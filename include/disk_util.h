#pragma once

#include "defaults.h"
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdlib.h>
#include <iostream>
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
inline void aligned_free(void *ptr)
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
struct QueryStats
{
    float total_us = 0; // total time to process query in micros
    float io_nbr_us = 0;    // total time spent in IO
    float io_vec_us = 0;    // total time spent in IO
    float cpu_us = 0;   // total time spent in CPU

    unsigned n_4k = 0;         // # of 4kB reads
    unsigned n_8k = 0;         // # of 8kB reads
    unsigned n_12k = 0;        // # of 12kB reads
    unsigned n_ios = 0;        // total # of IOs issued
    unsigned read_size = 0;    // total # of bytes read
    unsigned n_cmps_saved = 0; // # cmps saved
    unsigned n_cmps = 0;       // # cmps
    unsigned n_cache_hits = 0; // # cache_hits
    unsigned n_hops = 0;       // # search hops
};
class Timer
{
    typedef std::chrono::high_resolution_clock _clock;
    std::chrono::time_point<_clock> check_point;

  public:
    Timer() : check_point(_clock::now())
    {
    }

    void reset()
    {
        check_point = _clock::now();
    }

    long long elapsed() const
    {
        return std::chrono::duration_cast<std::chrono::microseconds>(_clock::now() - check_point).count();
    }

    float elapsed_seconds() const
    {
        return (float)elapsed() / 1000000.0f;
    }

    std::string elapsed_seconds_for_step(const std::string &step) const
    {
        return std::string("Time for ") + step + std::string(": ") + std::to_string(elapsed_seconds()) +
               std::string(" seconds");
    }
};

template <typename T>
inline double get_mean_stats(QueryStats *stats, uint64_t len, const std::function<T(const QueryStats &)> &member_fn)
{
    double avg = 0;
    for (uint64_t i = 0; i < len; i++)
    {
        avg += (double)member_fn(stats[i]);
    }
    return avg / len;
}
}