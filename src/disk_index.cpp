

#include "disk.h"
#include "disk_util.h"
#include "distance.h"
#include <boost/container_hash/detail/hash_range.hpp>
#include <cstdint>

namespace disk {
    void DiskIndex::init() {
        e_dist_ = new stkq::E_Distance(1);
        s_dist_ = new stkq::E_Distance(1);
        ctx_ = 0;
        int ret = io_setup(MAX_EVENTS, &ctx_);
        if (ret != 0)
        {
            if (ret == -EAGAIN)
            {
                std::cerr << "io_setup() failed with EAGAIN: Consider increasing /proc/sys/fs/aio-max-nr" << std::endl;
            }
            else
            {
                std::cerr << "io_setup() failed; returned " << ret << ": " << ::strerror(-ret) << std::endl;
            }
        }
    }

    void DiskIndex::load_metadata(const char *index_file) {
        std::cout<< "open file: " << index_file << std::endl;
        std::ifstream index_metadata(index_file, std::ios::binary);

        uint32_t node_num, max_alpha_range_len, max_nbr_len, ep_size, emb_dim, loc_dim;
        READ_U32(index_metadata, node_num);
        READ_U32(index_metadata, max_nbr_len);
        READ_U32(index_metadata, max_alpha_range_len);
        READ_U32(index_metadata, ep_size);
        READ_U32(index_metadata, emb_dim);
        READ_U32(index_metadata, loc_dim);

        size_t max_data_size = (emb_dim * sizeof(float)) + 
                                (loc_dim * sizeof(float)) +
                                sizeof(uint32_t) +
                                (max_nbr_len * (sizeof(uint32_t) + 2*max_alpha_range_len*sizeof(int8_t)));
        size_t max_aligned_size = disk::align_to_page_size(max_data_size);
        std::cout << "node size: " << node_num << std::endl;
        std::cout << "max aplha range len: " << max_alpha_range_len << std::endl;
        std::cout << "max neighbor len: " << max_nbr_len << std::endl;
        std::cout << "enter point size: " << ep_size << std::endl;
        std::cout << "emb dim: " << emb_dim << std::endl;
        std::cout << "loc dim: " << loc_dim << std::endl;
        std::cout<< "max data size: " << max_data_size << "B max aligned size: " << max_aligned_size << "B" << std::endl;
        _max_node_len = max_data_size;
        _max_nbr_len = max_nbr_len;
        _max_alpha_range_len = max_alpha_range_len;
        _num_points = node_num;
        _max_degree = max_nbr_len;
        emb_dim_ = emb_dim;
        loc_dim_ = loc_dim;
        enterpoint_set.reserve(ep_size);
        for (size_t i = 0; i < ep_size; i ++) {
            uint32_t id;
            READ_U32(index_metadata, id);
            enterpoint_set.emplace_back(id);
        }

        index_metadata.close();

        int flags = O_DIRECT | O_RDONLY | O_LARGEFILE;
        file_desc_ = open(index_file, flags);
        // error checks
        assert(this->file_desc_ != -1);
        std::cerr << "Opened file : " << index_file << std::endl;
        setup_sector_scratch();
    }
}