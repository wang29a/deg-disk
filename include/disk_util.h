#pragma once

#include "defaults.h"
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdlib.h>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>
#include <unistd.h> // For sysconf(_SC_PAGESIZE)

#define MAX_EVENTS 1024

namespace disk {
#define ROUND_UP(X, Y) ((((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0)) * (Y))

#define DIV_ROUND_UP(X, Y) (((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0))

// alignment tests
#define IS_ALIGNED(X, Y) ((uint64_t)(X) % (uint64_t)(Y) == 0)
#define IS_512_ALIGNED(X) IS_ALIGNED(X, 512)

#define READ_U64(stream, val) stream.read((char *)&val, sizeof(uint64_t))
#define READ_U32(stream, val) stream.read((char *)&val, sizeof(uint32_t))
#define READ_INT8(stream, val) stream.read((char *)&val, sizeof(int8_t))
#define READ_F32(stream, val) stream.read((char *)&val, sizeof(float))
// NOTE :: all 3 fields must be 512-aligned
struct AlignedRead
{
    uint64_t offset; // where to read from
    uint64_t len;    // how much to read
    void *buf;       // where to read into

    AlignedRead() : offset(0), len(0), buf(nullptr)
    {
    }

    ~AlignedRead() {
    }

    AlignedRead(uint64_t offset, uint64_t len, void *buf) : offset(offset), len(len), buf(buf)
    {
        assert(IS_512_ALIGNED(offset));
        assert(IS_512_ALIGNED(len));
        assert(IS_512_ALIGNED(buf));
        // assert(malloc_usable_size(buf) >= len);
    }
};

inline void alloc_aligned(void **ptr, size_t size, size_t align)
{
    *ptr = nullptr;
    *ptr = ::aligned_alloc(align, size);
}
inline void free_aligned(void *ptr)
{
    // Gopal. Must have a check here if the pointer was actually allocated by
    // _alloc_aligned
    if (ptr == nullptr)
    {
        return;
    }
    free(ptr);
}

inline size_t align_to_page_size(size_t data_size) {
    // 获取系统的内存页大小，通常为 4096
    // long page_size = sysconf(_SC_PAGESIZE);
    // if (page_size <= 0) {
    //     page_size = 4096; // 默认值
    // }
    long page_size = defaults::SECTOR_LEN;
    
    // 计算需要添加的填充字节数
    size_t padding_needed = (page_size - (data_size % page_size)) % page_size;
    
    // 返回页对齐后的总大小
    return data_size + padding_needed;
}

struct NbrData {
    uint32_t id;
    std::vector<int8_t> alpha_range;
};

struct NodeData {
    std::vector<float> emb;
    std::vector<float> loc;
    uint32_t nnbr;
    std::vector<NbrData> nbrs;
};

struct ScratchContext {
    float *emb_scratch = nullptr;
    float *loc_scratch = nullptr;
    char *sector_scratch = nullptr;
    
    size_t emb_size;
    size_t loc_size;
    size_t sector_size;
    size_t sector_idx;

    // 构造函数：负责分配内存
    ScratchContext(size_t emb_dim, size_t loc_dim) {
        emb_size = ROUND_UP(sizeof(float) * emb_dim, 256);
        loc_size = ROUND_UP(sizeof(float) * loc_dim, 256);
        // 假设 defaults::MAX_N_SECTOR_READS 和 SECTOR_LEN 是全局常量
        size_t max_sector_reads = 32; // 示例值
        size_t sector_len = 4096;     // 示例值
        sector_size = max_sector_reads * sector_len;

        alloc_aligned((void **)&emb_scratch, emb_size, 256);
        alloc_aligned((void **)&loc_scratch, loc_size, 256);
        alloc_aligned((void **)&sector_scratch, sector_size, sector_len);

        reset();
    }

    // 重置内存 (每次从池中取出时调用)
    void reset() {
        memset(emb_scratch, 0, emb_size);
        memset(loc_scratch, 0, loc_size);
        sector_idx = 0;
        // sector_scratch 通常作为读取缓冲区，可能不需要 memset，视需求而定
    }

    // 析构函数：负责释放内存
    ~ScratchContext() {
        if (emb_scratch) free_aligned(emb_scratch);
        if (loc_scratch) free_aligned(loc_scratch);
        if (sector_scratch) free_aligned(sector_scratch);
    }
    
    // 禁止拷贝，防止双重释放
    ScratchContext(const ScratchContext&) = delete;
    ScratchContext& operator=(const ScratchContext&) = delete;
};


class ScratchPool {
public:
    ScratchPool(size_t emb_dim, size_t loc_dim) 
        : emb_dim_(emb_dim), loc_dim_(loc_dim) {}

    // 显式清理池中所有内存（通常在 DiskIndex 析构时调用）
    void clear() {
        std::lock_guard<std::mutex> lock(mtx_);
        for (auto* ctx : pool_) {
            delete ctx;
        }
        pool_.clear();
    }

    ~ScratchPool() {
        clear();
    }


private:
    size_t emb_dim_;
    size_t loc_dim_;
    std::vector<ScratchContext*> pool_;
    std::mutex mtx_;

    // 辅助函数：创建带有“自动归还”功能的智能指针
    auto create_ptr(ScratchContext* ctx) {
        return std::unique_ptr<ScratchContext, std::function<void(ScratchContext*)>>(
            ctx,
            [this](ScratchContext* ptr) {
                std::lock_guard<std::mutex> lock(this->mtx_);
                this->pool_.push_back(ptr);
            }
        );
    }
public:
    // 获取一个可用的 Context
    std::unique_ptr<ScratchContext, std::function<void(ScratchContext*)>> acquire() {
        std::unique_lock<std::mutex> lock(mtx_);
        
        if (!pool_.empty()) {
            ScratchContext* ctx = pool_.back();
            pool_.pop_back();
            ctx->reset(); // 重置状态
            // 返回 unique_ptr，使用自定义删除器将对象归还给池，而不是 delete
            return create_ptr(ctx);
        }
        
        // 如果池为空，创建一个新的
        // 注意：这里释放锁，因为分配内存可能耗时
        lock.unlock(); 
        ScratchContext* new_ctx = new ScratchContext(emb_dim_, loc_dim_);
        return create_ptr(new_ctx);
    }

};

}