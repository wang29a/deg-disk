
#include "builder.h"
#include "disk_util.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <libaio.h>

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
        uint32_t max_aplha_range_len = 0;
        uint32_t max_nbr_len = 0;
        uint32_t enterpoint_set_size = final_index_->enterpoint_set.size();
        uint32_t emb_dim = final_index_->getBaseEmbDim();
        uint32_t loc_dim = final_index_->getBaseLocDim();

        std::cout << "node size: " << node_num << std::endl;
        std::cout << "max aplha range len: " << max_aplha_range_len << std::endl;
        std::cout << "max neighbor len: " << max_nbr_len << std::endl;
        std::cout << "enter point size: " << enterpoint_set_size << std::endl;
        std::cout << "emb dim: " << emb_dim << std::endl;
        std::cout << "loc dim: " << loc_dim << std::endl;

        for (unsigned i = 0; i < final_index_->getBaseLen(); i++)
        {
            unsigned neighbor_size = final_index_->DEG_nodes_[i]->GetSearchFriends().size();
            max_nbr_len = std::max(neighbor_size, max_nbr_len);
            for (unsigned k = 0; k < neighbor_size; k++)
            {
                Index::DEGSimpleNeighbor &neighbor = final_index_->DEG_nodes_[i]->GetSearchFriends()[k];
                std::vector<std::pair<int8_t, int8_t>> &use_range = neighbor.active_range;

                unsigned range_size = use_range.size();
                max_aplha_range_len = std::max(max_aplha_range_len, range_size);
            }
        }
        
        std::vector<uint32_t> enterpoint_set;
        enterpoint_set.reserve(enterpoint_set_size);
        for (unsigned i = 0; i < enterpoint_set_size; i++)
        {
            unsigned node_id = final_index_->enterpoint_set[i];
            enterpoint_set.emplace_back(node_id);
        }
        //meta data
        size_t raw_meta_data_size = 
            sizeof(node_num) + sizeof(max_nbr_len) + sizeof(max_aplha_range_len) +
            sizeof(enterpoint_set_size) + sizeof(emb_dim) + sizeof(loc_dim) + (sizeof(uint32_t)*enterpoint_set_size); 
        size_t aligned_size = disk::align_to_page_size(raw_meta_data_size);
        size_t padding_size = aligned_size - raw_meta_data_size;
        std::vector<char> buffer(aligned_size, 0);
        char* current_ptr = buffer.data();

        // 复制数据到缓冲区
        auto copy_data = [&](const void* src, size_t size) {
            std::memcpy(current_ptr, src, size);
            current_ptr += size;
        };

        copy_data(&node_num, sizeof(uint32_t));
        copy_data(&max_nbr_len, sizeof(uint32_t));
        copy_data(&max_aplha_range_len, sizeof(uint32_t));
        copy_data(&enterpoint_set_size, sizeof(uint32_t));
        copy_data(&emb_dim, sizeof(uint32_t));
        copy_data(&loc_dim, sizeof(uint32_t));
        copy_data(enterpoint_set.data(), sizeof(uint32_t)*enterpoint_set_size);

        out.write(buffer.data(), aligned_size);

        uint32_t nbr_data_size = sizeof(uint32_t)+2*max_aplha_range_len*sizeof(int8_t);

        // node data
        for (size_t i = 0; i < node_num; i ++) {
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
                data.nbrs[j].alpha_range.resize(max_aplha_range_len*2);
                for (size_t k = 0; k < range_size; k ++) {
                    int8_t x = use_range[k].first;
                    int8_t y = use_range[k].second;
                    data.nbrs[j].alpha_range[2*k] = x;
                    data.nbrs[j].alpha_range[2*k+1] = y;
                }
            }
            size_t raw_data_size = (data.emb.size() * sizeof(float)) + 
                           (data.loc.size() * sizeof(float)) +
                           sizeof(data.nnbr) +
                           (data.nbrs.size() * nbr_data_size);
            
            size_t aligned_size = disk::align_to_page_size(raw_data_size);
            size_t padding_size = aligned_size - raw_data_size;

            std::vector<char> buffer(aligned_size, 0);
            char* current_ptr = buffer.data();

            // 复制数据到缓冲区
            auto copy_data = [&](const void* src, size_t size) {
                std::memcpy(current_ptr, src, size);
                current_ptr += size;
            };
            // 复制向量数据
            copy_data(data.emb.data(), data.emb.size() * sizeof(float));
            copy_data(data.loc.data(), data.loc.size() * sizeof(float));

            copy_data(&data.nnbr, sizeof(data.nnbr));

            // 复制邻居数据
            copy_data(data.nbrs.data(), nbr_data_size);

            out.write(buffer.data(), aligned_size);
        }

        out.close();
        return this;
    }


}

namespace disk {
    typedef struct io_event io_event_t;
    typedef struct iocb iocb_t;
    void execute_io(io_context_t ctx, int fd, std::vector<AlignedRead> &read_reqs, uint64_t n_retries = 0)
    {
    #ifdef DEBUG
        for (auto &req : read_reqs)
        {
            assert(IS_ALIGNED(req.len, 512));
            // std::cout << "request:"<<req.offset<<":"<<req.len << std::endl;
            assert(IS_ALIGNED(req.offset, 512));
            assert(IS_ALIGNED(req.buf, 512));
            // assert(malloc_usable_size(req.buf) >= req.len);
        }
    #endif

        // break-up requests into chunks of size MAX_EVENTS each
        uint64_t n_iters = ROUND_UP(read_reqs.size(), MAX_EVENTS) / MAX_EVENTS;
        for (uint64_t iter = 0; iter < n_iters; iter++)
        {
            uint64_t n_ops = std::min((uint64_t)read_reqs.size() - (iter * MAX_EVENTS), (uint64_t)MAX_EVENTS);
            std::vector<iocb_t *> cbs(n_ops, nullptr);
            std::vector<io_event_t> evts(n_ops);
            std::vector<struct iocb> cb(n_ops);
            for (uint64_t j = 0; j < n_ops; j++)
            {
                io_prep_pread(cb.data() + j, fd, read_reqs[j + iter * MAX_EVENTS].buf, read_reqs[j + iter * MAX_EVENTS].len,
                            read_reqs[j + iter * MAX_EVENTS].offset);
            }

            // initialize `cbs` using `cb` array
            //

            for (uint64_t i = 0; i < n_ops; i++)
            {
                cbs[i] = cb.data() + i;
            }

            uint64_t n_tries = 0;
            while (n_tries <= n_retries)
            {
                // issue reads
                int64_t ret = io_submit(ctx, (int64_t)n_ops, cbs.data());
                // if requests didn't get accepted
                if (ret != (int64_t)n_ops)
                {
                    std::cerr << "io_submit() failed; returned " << ret << ", expected=" << n_ops << ", ernno=" << errno
                            << "=" << ::strerror(-ret) << ", try #" << n_tries + 1;
                    std::cout << "ctx: " << ctx << "\n";
                    exit(-1);
                }
                else
                {
                    // wait on io_getevents
                    ret = io_getevents(ctx, (int64_t)n_ops, (int64_t)n_ops, evts.data(), nullptr);
                    // if requests didn't complete
                    if (ret != (int64_t)n_ops)
                    {
                        std::cerr << "io_getevents() failed; returned " << ret << ", expected=" << n_ops
                                << ", ernno=" << errno << "=" << ::strerror(-ret) << ", try #" << n_tries + 1;
                        exit(-1);
                    }
                    else
                    {
                        break;
                    }
                }
            }
            // disabled since req.buf could be an offset into another buf
            /*
            for (auto &req : read_reqs) {
            // corruption check
            assert(malloc_usable_size(req.buf) >= req.len);
            }
            */
        }
    }
}
