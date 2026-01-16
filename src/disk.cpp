
#include "builder.h"
#include "disk_util.h"
#include "utils.h"
#include "disk.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <libaio.h>
#include <set>
#include <unordered_set>
#include <sys/types.h>

namespace stkq {
    /* TODO
       index formt
       meta data: vector1 dimension vector 2 dimension
       entry point ids
       node data
    */ 
    IndexBuilder *IndexBuilder::save_graph_disk(TYPE type, char *graph_file)
    {
        std::fstream out(graph_file, std::ios::binary | std::ios::out);
        // type == INDEX_DEG
        uint32_t node_num = final_index_->getBaseLen();
        uint32_t max_alpha_range_len = 0;
        uint32_t max_nbr_len = 0;
        uint32_t enterpoint_set_size = final_index_->DEG_enterpoints.size();
        uint32_t emb_dim = final_index_->getBaseEmbDim();
        uint32_t loc_dim = final_index_->getBaseLocDim();


        for (unsigned i = 0; i < final_index_->getBaseLen(); i++)
        {
            unsigned neighbor_size = final_index_->DEG_nodes_[i]->GetSearchFriends().size();
            max_nbr_len = std::max(neighbor_size, max_nbr_len);
            for (unsigned k = 0; k < neighbor_size; k++)
            {
                Index::DEGSimpleNeighbor &neighbor = final_index_->DEG_nodes_[i]->GetSearchFriends()[k];
                std::vector<std::pair<int8_t, int8_t>> &use_range = neighbor.active_range;

                unsigned range_size = use_range.size();
                max_alpha_range_len = std::max(max_alpha_range_len, range_size);
            }
        }
        
        std::vector<uint32_t> enterpoint_set;
        enterpoint_set.reserve(enterpoint_set_size);
        for (unsigned i = 0; i < enterpoint_set_size; i++)
        {
            unsigned node_id = final_index_->DEG_enterpoints[i]->GetId();
            enterpoint_set.emplace_back(node_id);
        }
        std::set<uint32_t> ep_set{enterpoint_set.begin(), enterpoint_set.end()};
        //meta data
        size_t raw_meta_data_size = 
            sizeof(node_num) + sizeof(max_nbr_len) + sizeof(max_alpha_range_len) +
            sizeof(enterpoint_set_size) + sizeof(emb_dim) + sizeof(loc_dim) + (sizeof(uint32_t)*enterpoint_set_size); 
        size_t aligned_size = disk::align_to_page_size(raw_meta_data_size);
        size_t padding_size = aligned_size - raw_meta_data_size;
        std::vector<char> buffer(aligned_size, 0);
        char* current_ptr = buffer.data();
        std::cout<<aligned_size << " metadata size" << std::endl;

        // 复制数据到缓冲区
        auto copy_data = [&](const void* src, size_t size) {
            std::memcpy(current_ptr, src, size);
            current_ptr += size;
        };

        copy_data(&node_num, sizeof(uint32_t));
        copy_data(&max_nbr_len, sizeof(uint32_t));
        copy_data(&max_alpha_range_len, sizeof(uint32_t));
        copy_data(&enterpoint_set_size, sizeof(uint32_t));
        copy_data(&emb_dim, sizeof(uint32_t));
        copy_data(&loc_dim, sizeof(uint32_t));
        copy_data(enterpoint_set.data(), sizeof(uint32_t)*enterpoint_set_size);

        out.write(buffer.data(), aligned_size);

        uint32_t nbr_data_size = sizeof(uint32_t)+2*max_alpha_range_len*sizeof(int8_t);

        // node data
        size_t max_data_size = (emb_dim * sizeof(float)) + 
                                (loc_dim * sizeof(float)) +
                                sizeof(uint32_t) +
                                (max_nbr_len * (sizeof(uint32_t) + 2*max_alpha_range_len*sizeof(int8_t)));
        size_t max_aligned_size = disk::align_to_page_size(max_data_size);
        std::cout << "node size: " << node_num << std::endl;
        std::cout << "max aplha range len: " << max_alpha_range_len << std::endl;
        std::cout << "max neighbor len: " << max_nbr_len << std::endl;
        std::cout << "enter point size: " << enterpoint_set_size << std::endl;
        std::cout << "emb dim: " << emb_dim << std::endl;
        std::cout << "loc dim: " << loc_dim << std::endl;
        std::cout<< "max data size: " << max_data_size << "B max aligned size: " << max_aligned_size << "B" << std::endl;

        for (size_t i = 0; i < node_num; i ++) {
            if (ep_set.find(i) != ep_set.end()) {
                std::cout<< "ep: " << i << " offset: " << static_cast<std::size_t>(out.tellp()) << std::endl;
            }
            disk::NodeData data;
            data.emb.resize(emb_dim);
            data.loc.resize(loc_dim);

            memcpy(data.emb.data(), final_index_->getBaseEmbData()+i*emb_dim, emb_dim*sizeof(float));
            memcpy(data.loc.data(), final_index_->getBaseLocData()+i*loc_dim, loc_dim*sizeof(float));

            unsigned neighbor_size = final_index_->DEG_nodes_[i]->GetSearchFriends().size();
            data.nnbr = neighbor_size;
            data.nbrs.resize(neighbor_size);

            for (size_t j = 0; j < neighbor_size; j ++) {
                Index::DEGSimpleNeighbor &neighbor = final_index_->DEG_nodes_[i]->GetSearchFriends()[j];
                unsigned neighbor_id = neighbor.id_;
                data.nbrs[j].id = neighbor_id;
                std::vector<std::pair<int8_t, int8_t>> &use_range = neighbor.active_range;

                unsigned range_size = use_range.size();
                data.nbrs[j].alpha_range.resize(max_alpha_range_len*2);
                for (size_t k = 0; k < range_size; k ++) {
                    int8_t x = use_range[k].first;
                    int8_t y = use_range[k].second;
                    data.nbrs[j].alpha_range[2*k] = x;
                    data.nbrs[j].alpha_range[2*k+1] = y;
                }
            }
            size_t raw_data_size = (emb_dim * sizeof(float)) + 
                           (loc_dim * sizeof(float)) +
                           sizeof(data.nnbr) +
                           (data.nbrs.size() * nbr_data_size);
            assert(raw_data_size <= max_data_size);
            size_t padding_size = max_aligned_size - raw_data_size;

            std::vector<char> buffer(max_aligned_size, 0);
            char* current_ptr = buffer.data();

            // 复制数据到缓冲区
            auto copy_data = [&](const void* src, size_t size) {
                std::memcpy(current_ptr, src, size);
                current_ptr += size;
            };
            // 复制向量数据
            copy_data(data.emb.data(), data.emb.size() * sizeof(float));
            copy_data(data.loc.data(), data.loc.size() * sizeof(float));
            // if (ep_set.find(i) != ep_set.end()) {
            //     for (size_t k = 0; k < 20; k ++) {
            //         std::cout<< *(float*)(buffer.data()+(size_t)k*sizeof(float)) << " ";
            //     }
            //     std::cout<<std::endl;
            //     for (size_t k = 0; k < 20; k ++) {
            //         std::cout<< *(float*)(buffer.data()+(size_t)k*sizeof(float)+(size_t)emb_dim*sizeof(float)) << " ";
            //         std::cout<< data.loc[k] << " ";
            //     }
            //     std::cout<<std::endl;
            // }

            copy_data(&data.nnbr, sizeof(data.nnbr));

            // 复制邻居数据
            for (auto &nbr : data.nbrs) {
                copy_data(&nbr.id, sizeof(uint32_t));
                copy_data(nbr.alpha_range.data(), nbr.alpha_range.size());
            }

            out.write(buffer.data(), max_aligned_size);
        }

        out.close();
        return this;
    }
    // 修改函数签名，接收三个文件名
    IndexBuilder *IndexBuilder::save_graph_disk(TYPE type, char *meta_file, char *graph_file, char *data_file)
    {
        // [Image of vector index file structure showing separate metadata, topology, and vector data files]
        
        std::fstream out_meta(meta_file, std::ios::binary | std::ios::out);
        std::fstream out_graph(graph_file, std::ios::binary | std::ios::out);
        std::fstream out_data(data_file, std::ios::binary | std::ios::out);

        if (!out_meta.is_open() || !out_graph.is_open() || !out_data.is_open()) {
            std::cerr << "Error opening output files." << std::endl;
            return this;
        }

        const size_t PAGE_SIZE = 8192; // 定义页大小

        // 1. 基础信息统计 & 预计算最大值
        uint32_t node_num = final_index_->getBaseLen();
        uint32_t max_alpha_range_len = 0;
        uint32_t max_nbr_len = 0;
        uint32_t enterpoint_set_size = final_index_->DEG_enterpoints.size();
        uint32_t emb_dim = final_index_->getBaseEmbDim();
        uint32_t loc_dim = final_index_->getBaseLocDim();

        for (unsigned i = 0; i < node_num; i++) {
            unsigned neighbor_size = final_index_->DEG_nodes_[i]->GetFriends().size();
            max_nbr_len = std::max(neighbor_size, max_nbr_len);
            for (unsigned k = 0; k < neighbor_size; k++) {
                Index::DEGNeighbor &neighbor = final_index_->DEG_nodes_[i]->GetFriends()[k];
                max_alpha_range_len = std::max(max_alpha_range_len, (uint32_t)neighbor.available_range.size());
            }
        }

        std::vector<uint32_t> enterpoint_set;
        enterpoint_set.reserve(enterpoint_set_size);
        for (unsigned i = 0; i < enterpoint_set_size; i++) {
            enterpoint_set.emplace_back(final_index_->DEG_enterpoints[i]->GetId());
        }

        uint32_t single_neighbor_size = sizeof(uint32_t) + (2 * max_alpha_range_len * sizeof(int8_t));
        size_t fixed_topo_size = sizeof(uint32_t) + (max_nbr_len * single_neighbor_size);
        uint64_t nnodes_per_sector = PAGE_SIZE / fixed_topo_size;
        // ==========================================
        // Part 1: 写入 Meta Data (一次性写入即可，通常较小)
        // ==========================================
        // 即使是 Meta，为了对齐习惯，我们也补齐到 PageSize (可选，但为了规范建议做)
        size_t raw_meta_size = 
            sizeof(node_num) + sizeof(max_nbr_len) + sizeof(max_alpha_range_len) +
            sizeof(enterpoint_set_size) + sizeof(emb_dim) + sizeof(loc_dim) +sizeof(nnodes_per_sector) + 
            (sizeof(uint32_t) * enterpoint_set_size); 

        size_t aligned_meta_size = disk::align_to_page_size(raw_meta_size);
        std::vector<char> meta_buffer(aligned_meta_size, 0);
        char* meta_ptr = meta_buffer.data();

        auto copy_to_ptr = [&](char*& ptr, const void* src, size_t size) {
            std::memcpy(ptr, src, size);
            ptr += size;
        };

        copy_to_ptr(meta_ptr, &node_num, sizeof(uint32_t));
        copy_to_ptr(meta_ptr, &emb_dim, sizeof(uint32_t));
        copy_to_ptr(meta_ptr, &loc_dim, sizeof(uint32_t));
        copy_to_ptr(meta_ptr, &max_nbr_len, sizeof(uint32_t));
        copy_to_ptr(meta_ptr, &max_alpha_range_len, sizeof(uint32_t));
        copy_to_ptr(meta_ptr, &nnodes_per_sector, sizeof(uint64_t));
        copy_to_ptr(meta_ptr, &enterpoint_set_size, sizeof(uint32_t));
        copy_to_ptr(meta_ptr, enterpoint_set.data(), sizeof(uint32_t) * enterpoint_set_size);
        std::cout << "node size: " << node_num << std::endl;
        std::cout << "emb dim: " << emb_dim << std::endl;
        std::cout << "loc dim: " << loc_dim << std::endl;
        std::cout << "max aplha range len: " << max_alpha_range_len << std::endl;
        std::cout << "max neighbor len: " << max_nbr_len << std::endl;
        std::cout << "nnodes_per_sector: " << nnodes_per_sector << std::endl;
        std::cout << "enter point size: " << enterpoint_set_size << std::endl;

        out_meta.write(meta_buffer.data(), aligned_meta_size);
        out_meta.close();
        std::cout << "Metadata saved. Aligned size: " << aligned_meta_size << std::endl;


        // ==========================================
        // 通用 Buffer 写入 Lambda (核心逻辑)
        // [Image of buffer flushing mechanism to disk storage]
        // ==========================================
        // 参数: 文件流, Buffer vector, 当前Buffer偏移量(引用), 数据源, 数据长度
        auto buffered_write = [&](std::fstream& fs, std::vector<char>& buf, size_t& buf_off, const void* src, size_t size) {
            const char* src_ptr = (const char*)src;
            size_t written = 0;
            while (written < size) {
                size_t space_left = PAGE_SIZE - buf_off;
                size_t chunk = std::min(space_left, size - written);
                
                std::memcpy(buf.data() + buf_off, src_ptr + written, chunk);
                buf_off += chunk;
                written += chunk;

                // 满页落盘
                if (buf_off == PAGE_SIZE) {
                    fs.write(buf.data(), PAGE_SIZE);
                    buf_off = 0; // 重置偏移量
                    // std::fill(buf.begin(), buf.end(), 0); // 性能优化可省略fill，因为会被覆盖
                }
            }
        };
        auto buffered_write_aligned = [&](std::fstream& fs, std::vector<char>& buf, size_t& buf_off, const void* src, size_t size) {
            // 1. 安全检查：如果单条数据比一整页还大，这种逻辑是无法处理的
            if (size > PAGE_SIZE) {
                throw std::runtime_error("Data too large for a single page");
            }

            // 2. 判断当前页是否放得下
            if (buf_off + size > PAGE_SIZE) {
                // 放不下 -> 也就是“当前写入超过page”
                
                // A. 将当前页剩余空间补 0 (Padding)
                std::memset(buf.data() + buf_off, 0, PAGE_SIZE - buf_off);
                
                // B. 前一个 Page 落盘
                fs.write(buf.data(), PAGE_SIZE);
                
                // C. 开一个新的 Page (重置偏移)
                buf_off = 0;
            }

            // 3. 将数据写入 Buffer (此时 Buffer 空间一定足够)
            std::memcpy(buf.data() + buf_off, src, size);
            buf_off += size;
        };

        // 结束时的 Flush Lambda (处理不足一页的数据)
        auto final_flush = [&](std::fstream& fs, std::vector<char>& buf, size_t& buf_off) {
            if (buf_off > 0) {
                // 剩余部分补 0
                std::memset(buf.data() + buf_off, 0, PAGE_SIZE - buf_off);
                fs.write(buf.data(), PAGE_SIZE);
            }
        };

        // ==========================================
        // Part 2: 写入 Vector Data (向量数据)
        // ==========================================
        std::vector<char> data_page_buf(PAGE_SIZE, 0);
        size_t data_buf_off = 0;
        size_t single_node_data_size = (emb_dim + loc_dim) * sizeof(float);

        std::cout << "Writing Vector Data... Node Size: " << single_node_data_size << " B" << std::endl;

        for (size_t i = 0; i < node_num; i++) {
            // 写入 Emb
            buffered_write_aligned(out_data, data_page_buf, data_buf_off, 
                        final_index_->getBaseEmbData() + i * emb_dim, emb_dim * sizeof(float));
            // 写入 Loc
            buffered_write_aligned(out_data, data_page_buf, data_buf_off, 
                        final_index_->getBaseLocData() + i * loc_dim, loc_dim * sizeof(float));
        }
        // 处理最后一个 Page
        final_flush(out_data, data_page_buf, data_buf_off);
        out_data.close();


        // ==========================================
        // Part 3: 写入 Graph Topology (拓扑数据)
        // ==========================================
        std::vector<char> graph_page_buf(PAGE_SIZE, 0);
        size_t graph_buf_off = 0;

        // 计算单个节点的固定拓扑大小 (Fixed Size)
        // 结构: [Nbr Count (4B)] + [Nbr_1 ID] [Nbr_1 Range] ... [Nbr_Max ID] [Nbr_Max Range]
        // 即使邻居不够 Max 个，也要预留空间填 0
        single_neighbor_size = sizeof(uint32_t) + (2 * max_alpha_range_len * sizeof(int8_t));
        fixed_topo_size = sizeof(uint32_t) + (max_nbr_len * single_neighbor_size);

        std::cout << "Writing Graph Topology... Fixed Node Size: " << fixed_topo_size << " B" << std::endl;

        // 临时 Buffer，用于构建当前节点的完整数据，然后再喂给 buffered_write
        // 这样做是为了方便处理 Padding 0
        std::vector<char> node_topo_buffer(fixed_topo_size, 0); 

        std::unordered_set<uint32_t> ep_set{enterpoint_set.begin(), enterpoint_set.end()};
        for (size_t i = 0; i < node_num; i++) {
            // if (ep_set.find(i) != ep_set.end()) {
            //     std::cout<< "ep: " << i << " offset: " << static_cast<std::size_t>(out_graph.tellp()) << " :" << i/6*PAGE_SIZE << std::endl;
            // }
            // 1. 清空当前节点 Buffer (全部置 0，相当于完成了 Padding)
            std::memset(node_topo_buffer.data(), 0, fixed_topo_size);
            
            char* ptr = node_topo_buffer.data();
            unsigned neighbor_size = final_index_->DEG_nodes_[i]->GetFriends().size();

            // 2. 写入实际数据
            // 2.1 写入邻居数量
            std::memcpy(ptr, &neighbor_size, sizeof(uint32_t));
            ptr += sizeof(uint32_t);

            // 2.2 写入存在的邻居
            for (unsigned k = 0; k < neighbor_size; k++) {
                Index::DEGNeighbor &neighbor = final_index_->DEG_nodes_[i]->GetFriends()[k];
                
                // 写入 ID
                std::memcpy(ptr, &neighbor.id_, sizeof(uint32_t));
                ptr += sizeof(uint32_t);

                // 写入 Range
                auto &use_range = neighbor.available_range;
                unsigned range_size = use_range.size();
                
                // 即使 range 不满 max_alpha_range_len，我们也按顺序写，剩下的已经在 memset 0 时处理了
                // 注意：这里需要按照 max_alpha_range_len 的总长度来跳过指针，保持定长结构
                
                size_t range_bytes_used = 0;
                for(size_t r = 0; r < range_size; ++r) {
                    // 将 pair 拆解写入
                    int8_t x = static_cast<int8_t>(use_range[r].first * 100);
                    int8_t y = static_cast<int8_t>(use_range[r].second * 100);
                    std::memcpy(ptr, &x, sizeof(int8_t)); ptr += sizeof(int8_t);
                    std::memcpy(ptr, &y, sizeof(int8_t)); ptr += sizeof(int8_t);
                    range_bytes_used += 2 * sizeof(int8_t);
                }

                // 跳过剩余的 Range Padding 空间，指向下一个邻居的起始位置
                size_t range_total_bytes = 2 * max_alpha_range_len * sizeof(int8_t);
                ptr += (range_total_bytes - range_bytes_used);
            }

            // 此时 node_topo_buffer 中前部是真实数据，后部是 0，且总长度为 fixed_topo_size
            // 3. 将固定大小的节点数据写入 Page Buffer
            buffered_write_aligned(out_graph, graph_page_buf, graph_buf_off, node_topo_buffer.data(), fixed_topo_size);

            if (i % 100000 == 0) std::cout << "Saved topology " << i << " nodes..." << std::endl;
        }

        // 处理最后一个 Page
        final_flush(out_graph, graph_page_buf, graph_buf_off);
        out_graph.close();

        return this;
    }
    IndexBuilder *IndexBuilder::save_graph_disk_decouple(TYPE type, char *graph_file)
    {
        std::fstream out(graph_file, std::ios::binary | std::ios::out);
        // type == INDEX_DEG
        uint32_t node_num = final_index_->getBaseLen();
        uint32_t max_alpha_range_len = 0;
        uint32_t max_nbr_len = 0;
        uint32_t enterpoint_set_size = final_index_->enterpoint_set.size();
        uint32_t emb_dim = final_index_->getBaseEmbDim();
        uint32_t loc_dim = final_index_->getBaseLocDim();

        for (unsigned i = 0; i < final_index_->getBaseLen(); i++)
        {
            unsigned neighbor_size = final_index_->DEG_nodes_[i]->GetSearchFriends().size();
            max_nbr_len = std::max(neighbor_size, max_nbr_len);
            for (unsigned k = 0; k < neighbor_size; k++)
            {
                Index::DEGSimpleNeighbor &neighbor = final_index_->DEG_nodes_[i]->GetSearchFriends()[k];
                std::vector<std::pair<int8_t, int8_t>> &use_range = neighbor.active_range;

                unsigned range_size = use_range.size();
                max_alpha_range_len = std::max(max_alpha_range_len, range_size);
            }
        }
        
        std::vector<uint32_t> enterpoint_set;
        enterpoint_set.reserve(enterpoint_set_size);
        for (unsigned i = 0; i < enterpoint_set_size; i++)
        {
            unsigned node_id = final_index_->enterpoint_set[i];
            enterpoint_set.emplace_back(node_id);
        }
        std::set<uint32_t> ep_set{enterpoint_set.begin(), enterpoint_set.end()};
        //meta data
        size_t raw_meta_data_size = 
            sizeof(node_num) + sizeof(max_nbr_len) + sizeof(max_alpha_range_len) +
            sizeof(enterpoint_set_size) + sizeof(emb_dim) + sizeof(loc_dim) + (sizeof(uint32_t)*enterpoint_set_size); 
        size_t aligned_size = disk::align_to_page_size(raw_meta_data_size);
        size_t padding_size = aligned_size - raw_meta_data_size;
        std::vector<char> buffer(aligned_size, 0);
        char* current_ptr = buffer.data();
        std::cout<<aligned_size << " metadata size" << std::endl;

        // 复制数据到缓冲区
        auto copy_data = [&](const void* src, size_t size) {
            std::memcpy(current_ptr, src, size);
            current_ptr += size;
        };

        copy_data(&node_num, sizeof(uint32_t));
        copy_data(&emb_dim, sizeof(uint32_t));
        copy_data(&loc_dim, sizeof(uint32_t));
        copy_data(&max_nbr_len, sizeof(uint32_t));
        copy_data(&max_alpha_range_len, sizeof(uint32_t));
        copy_data(&enterpoint_set_size, sizeof(uint32_t));
        copy_data(enterpoint_set.data(), sizeof(uint32_t)*enterpoint_set_size);

        out.write(buffer.data(), aligned_size);

        uint32_t nbr_data_size = sizeof(uint32_t)+2*max_alpha_range_len*sizeof(int8_t);

        // node data
        size_t max_data_size = (emb_dim * sizeof(float)) + 
                                (loc_dim * sizeof(float)) +
                                sizeof(uint32_t) +
                                (max_nbr_len * (sizeof(uint32_t) + 2*max_alpha_range_len*sizeof(int8_t)));
        size_t max_aligned_size = disk::align_to_page_size(max_data_size);
        std::cout << "node size: " << node_num << std::endl;
        std::cout << "emb dim: " << emb_dim << std::endl;
        std::cout << "loc dim: " << loc_dim << std::endl;
        std::cout << "max aplha range len: " << max_alpha_range_len << std::endl;
        std::cout << "max neighbor len: " << max_nbr_len << std::endl;
        std::cout << "enter point size: " << enterpoint_set_size << std::endl;
        std::cout<< "max data size: " << max_data_size << "B max aligned size: " << max_aligned_size << "B" << std::endl;
        for (size_t i = 0; i < node_num; i ++) {
            // if (ep_set.find(i) != ep_set.end()) {
            //     std::cout<< "ep: " << i << " offset: " << static_cast<std::size_t>(out.tellp()) << std::endl;
            // }
            disk::NodeData data;
            data.emb.resize(emb_dim);
            data.loc.resize(loc_dim);

            memcpy(data.emb.data(), final_index_->getBaseEmbData()+i*emb_dim, emb_dim*sizeof(float));
            memcpy(data.loc.data(), final_index_->getBaseLocData()+i*loc_dim, loc_dim*sizeof(float));

            unsigned neighbor_size = final_index_->DEG_nodes_[i]->GetSearchFriends().size();
            data.nnbr = neighbor_size;
            data.nbrs.resize(neighbor_size);

            for (size_t j = 0; j < neighbor_size; j ++) {
                Index::DEGSimpleNeighbor &neighbor = final_index_->DEG_nodes_[i]->GetSearchFriends()[j];
                unsigned neighbor_id = neighbor.id_;
                data.nbrs[j].id = neighbor_id;
                std::vector<std::pair<int8_t, int8_t>> &use_range = neighbor.active_range;

                unsigned range_size = use_range.size();
                data.nbrs[j].alpha_range.resize(max_alpha_range_len*2);
                for (size_t k = 0; k < range_size; k ++) {
                    int8_t x = use_range[k].first;
                    int8_t y = use_range[k].second;
                    data.nbrs[j].alpha_range[2*k] = x;
                    data.nbrs[j].alpha_range[2*k+1] = y;
                }
            }
            size_t raw_data_size = (emb_dim * sizeof(float)) + 
                           (loc_dim * sizeof(float)) +
                           sizeof(data.nnbr) +
                           (data.nbrs.size() * nbr_data_size);
            assert(raw_data_size <= max_data_size);
            size_t padding_size = max_aligned_size - raw_data_size;

            std::vector<char> buffer(max_aligned_size, 0);
            char* current_ptr = buffer.data();

            // 复制数据到缓冲区
            auto copy_data = [&](const void* src, size_t size) {
                std::memcpy(current_ptr, src, size);
                current_ptr += size;
            };
            // 复制向量数据
            copy_data(data.emb.data(), data.emb.size() * sizeof(float));
            copy_data(data.loc.data(), data.loc.size() * sizeof(float));
            // if (ep_set.find(i) != ep_set.end()) {
            //     for (size_t k = 0; k < 20; k ++) {
            //         std::cout<< *(float*)(buffer.data()+(size_t)k*sizeof(float)) << " ";
            //     }
            //     std::cout<<std::endl;
            //     for (size_t k = 0; k < 20; k ++) {
            //         std::cout<< *(float*)(buffer.data()+(size_t)k*sizeof(float)+(size_t)emb_dim*sizeof(float)) << " ";
            //         std::cout<< data.loc[k] << " ";
            //     }
            //     std::cout<<std::endl;
            // }

            copy_data(&data.nnbr, sizeof(data.nnbr));

            // 复制邻居数据
            for (auto &nbr : data.nbrs) {
                copy_data(&nbr.id, sizeof(uint32_t));
                copy_data(nbr.alpha_range.data(), nbr.alpha_range.size());
            }

            out.write(buffer.data(), max_aligned_size);
        }

        out.close();
        return this;
    }

}

namespace disk {
    typedef struct io_event io_event_t;
    typedef struct iocb iocb_t;
    void execute_io(io_context_t ctx, int fd, std::vector<AlignedRead> &read_reqs, uint64_t n_retries)
    {
    #ifdef DEBUG
        for (auto &req : read_reqs)
        {
            assert(IS_ALIGNED_(req.len, 512));
            // std::cout << "request:"<<req.offset<<":"<<req.len << std::endl;
            assert(IS_ALIGNED_(req.offset, 512));
            assert(IS_ALIGNED_(req.buf, 512));
            // assert(malloc_usable_size(req.buf) >= req.len);
        }
    #endif

        // break-up requests into chunks of size MAX_EVENTS each
        uint64_t n_iters = ROUND_UP(read_reqs.size(), MAX_EVENTS) / MAX_EVENTS;
        for (uint64_t iter = 0; iter < n_iters; iter++)
        {
            // uint64_t n_ops = std::min((uint64_t)read_reqs.size() - (iter * MAX_EVENTS), (uint64_t)MAX_EVENTS);
            // std::vector<iocb_t *> cbs(n_ops, nullptr);
            // std::vector<io_event_t> evts(n_ops);
            // std::vector<struct iocb> cb(n_ops);
            // for (uint64_t j = 0; j < n_ops; j++)
            // {
            //     io_prep_pread(cb.data() + j, fd, read_reqs[j + iter * MAX_EVENTS].buf, read_reqs[j + iter * MAX_EVENTS].len,
            //                 read_reqs[j + iter * MAX_EVENTS].offset);
            // }

            // // initialize `cbs` using `cb` array
            // //

            // for (uint64_t i = 0; i < n_ops; i++)
            // {
            //     cbs[i] = cb.data() + i;
            // }

            // uint64_t n_tries = 0;
            // while (n_tries <= n_retries)
            // {
            //     // issue reads
            //     int64_t ret = io_submit(ctx, (int64_t)n_ops, cbs.data());
            //     // if requests didn't get accepted
            //     if (ret != (int64_t)n_ops)
            //     {
            //         std::cerr << "io_submit() failed; returned " << ret << ", expected=" << n_ops << ", ernno=" << errno
            //                 << "=" << ::strerror(-ret) << ", try #" << n_tries + 1;
            //         std::cout << "ctx: " << ctx << "\n";
            //         exit(-1);
            //     }
            //     else
            //     {
            //         // wait on io_getevents
            //         ret = io_getevents(ctx, (int64_t)n_ops, (int64_t)n_ops, evts.data(), nullptr);
            //         // if requests didn't complete
            //         if (ret != (int64_t)n_ops)
            //         {
            //             std::cerr << "io_getevents() failed; returned " << ret << ", expected=" << n_ops
            //                     << ", ernno=" << errno << "=" << ::strerror(-ret) << ", try #" << n_tries + 1;
            //             exit(-1);
            //         }
            //         else
            //         {
            //             break;
            //         }
            //     }
            // }
            // // disabled since req.buf could be an offset into another buf
            // /*
            // for (auto &req : read_reqs) {
            // // corruption check
            // assert(malloc_usable_size(req.buf) >= req.len);
            // }
            // */
        }
    }

    template <typename T>
    inline void load_data(const char *filename, T *&data, unsigned &num, unsigned &dim)
    {
        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open())
        {
            std::cerr << "Error opening file " << filename << std::endl;
            exit(-1);
        }

        // 读取维度信息
        in.read((char *)&dim, 4);
        if (in.fail())
        {
            std::cerr << "Error reading dimension from file " << filename << std::endl;
            exit(-1);
        }

        // 获取文件大小
        in.seekg(0, std::ios::end);
        std::ios::pos_type ss = in.tellg();
        auto f_size = (size_t)ss;

        // 计算数据数量
        num = (unsigned)(f_size / (dim + 1) / 4);

        size_t total_size = (size_t)num * dim;
        // 分配内存
        try
        {
            data = new T[total_size];
        }
        catch (std::bad_alloc &)
        {
            std::cerr << "Memory allocation failed for data in " << filename << std::endl;
            exit(-1);
        }
        in.seekg(0, std::ios::beg);
        // 分块读取数据
        const size_t block_size = 10000 * dim; // 每次读取10000个数据块，可以根据需要调整
        size_t offset = 0;

        while (offset < total_size)
        {
            size_t remaining = total_size - offset;
            size_t current_block_size = std::min(block_size, remaining);

            for (size_t i = 0; i < current_block_size / dim; ++i)
            {
                // 读取并验证维度信息
                unsigned current_dim;
                in.read(reinterpret_cast<char *>(&current_dim), sizeof(current_dim));
                if (in.fail() || current_dim != dim)
                {
                    std::cerr << "Error reading dimension or dimension mismatch in file " << filename << " at index " << (offset / dim + i) << std::endl;
                    delete[] data;
                    exit(-1);
                }

                in.read(reinterpret_cast<char *>(data + offset + i * dim), dim * sizeof(T));
                if (in.fail())
                {
                    std::cerr << "Error reading data from file " << filename << " at index " << (offset / dim + i) << std::endl;
                    delete[] data;
                    exit(-1);
                }
            }

            offset += current_block_size;
        }

        in.close();
        // 输出调试信息
        std::cout << "Loaded " << num << " entries from " << filename << " with dimension " << dim << std::endl;
    }

    void QueryData::load(char *query_emb_file, char *query_loc_file, char *query_alpha_file, char *ground_file, stkq::Parameters &parameters) {
        float *query_emb = nullptr;
        unsigned query_num{};
        unsigned query_emb_dim{};
        load_data<float>(query_emb_file, query_emb, query_num, query_emb_dim);
        setQueryEmbData(query_emb);
        setQueryLen(query_num);
        setQueryEmbDim(query_emb_dim);
        assert(getQueryEmbData() != nullptr && getQueryLen() != 0 && getQueryEmbDim() != 0);
        float *query_loc = nullptr;
        unsigned query_loc_num{};
        unsigned query_loc_dim{};
        load_data(query_loc_file, query_loc, query_loc_num, query_loc_dim);
        setQueryLocData(query_loc);
        setQueryLocDim(query_loc_dim);
        assert(query_loc_num == getQueryLen());
        float *query_alpha = nullptr;
        unsigned query_alpha_num{};
        unsigned query_alpha_dim{};
        load_data(query_alpha_file, query_alpha, query_alpha_num, query_alpha_dim);
        setQueryWeightData(query_alpha);
        assert(query_loc_num == getQueryLen());
        unsigned *ground_data = nullptr;
        unsigned ground_num{};
        unsigned ground_dim{};
        load_data<unsigned>(ground_file, ground_data, ground_num, ground_dim);
        setGroundData(ground_data);
        setGroundLen(ground_num);
        setGroundDim(ground_dim);
        assert(getGroundData() != nullptr && getGroundLen() != 0 && getGroundDim() != 0);
        std::cout << "query data len : " << getQueryLen() << std::endl;
        std::cout << "query data emb dim : " << getQueryEmbDim() << std::endl;
        std::cout << "query data loc dim : " << getQueryLocDim() << std::endl;
        std::cout << "ground truth data len : " << getGroundLen() << std::endl;
        std::cout << "ground truth data dim : " << getGroundDim() << std::endl;
        std::cout << "=====================" << std::endl;
    }
}
