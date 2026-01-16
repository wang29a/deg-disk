

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
        scratch_pool_ = std::make_unique<ScratchPool>(emb_dim_, loc_dim_);
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
        // setup_sector_scratch();
    }

    void DiskIndex::load_graph_disk(char *graph_file)
    {
        std::fstream in(graph_file, std::ios::binary | std::ios::in);
        if (!in.is_open()) {
            std::cerr << "Error: Cannot open graph file " << graph_file << std::endl;
            return;
        }

        // --- 1. 读取元数据 (Metadata) ---

        uint32_t node_num = 0;
        uint32_t max_nbr_len = 0;
        uint32_t max_alpha_range_len = 0;
        uint32_t enterpoint_set_size = 0;
        uint32_t emb_dim = 0;
        uint32_t loc_dim = 0;

        // 先读取固定大小的头部变量
        // 注意：写入时是整个buffer一起写的，读取时可以顺序读，但必须处理对齐偏移
        
        // 为了准确跳过Padding，我们需要模拟写入时的内存布局计算
        // 写入顺序: node_num, max_nbr, max_alpha, ep_size, emb_dim, loc_dim, ep_vector
        
        size_t header_fixed_size = sizeof(uint32_t) * 6;
        std::vector<char> header_buffer(header_fixed_size);
        in.read(header_buffer.data(), header_fixed_size);

        char* ptr = header_buffer.data();
        auto read_uint32 = [&ptr]() -> uint32_t {
            uint32_t val;
            std::memcpy(&val, ptr, sizeof(uint32_t));
            ptr += sizeof(uint32_t);
            return val;
        };

        node_num = read_uint32();
        max_nbr_len = read_uint32();
        max_alpha_range_len = read_uint32();
        enterpoint_set_size = read_uint32();
        emb_dim = read_uint32();
        loc_dim = read_uint32();

        std::cout << "Loading graph..." << std::endl;
        std::cout << "node_num: " << node_num << std::endl;
        std::cout << "max_nbr_len: " << max_nbr_len << std::endl;
        std::cout << "max_alpha_range_len: " << max_alpha_range_len << std::endl;
        std::cout << "dims: " << emb_dim << ", " << loc_dim << std::endl;

        // 读取入口点集合
        std::vector<uint32_t> enterpoint_set(enterpoint_set_size);
        if (enterpoint_set_size > 0) {
            in.read(reinterpret_cast<char*>(enterpoint_set.data()), sizeof(uint32_t) * enterpoint_set_size);
        }

        // 计算元数据区的对齐大小，以便跳转到 Node Data 区
        size_t raw_meta_data_size = header_fixed_size + (sizeof(uint32_t) * enterpoint_set_size);
        size_t aligned_meta_size = disk::align_to_page_size(raw_meta_data_size);
        
        // 跳转到 Node Data 的起始位置 (跳过 Metadata 的 Padding)
        in.seekg(aligned_meta_size, std::ios::beg);

        // --- 2. 初始化索引结构 (根据你的类结构调整) ---
        
        // 假设 final_index_ 已经被分配，或者在这里进行初始化
        // final_index_->init(node_num, emb_dim, loc_dim); 
        // final_index_->DEG_enterpoints = ...; 

        // --- 3. 计算节点数据块大小 ---
        
        // 单个邻居数据块的大小 (ID + alpha_range pairs)
        // 写入逻辑: sizeof(uint32_t) + 2 * max_alpha_range_len * sizeof(int8_t)
        uint32_t nbr_data_size = sizeof(uint32_t) + 2 * max_alpha_range_len * sizeof(int8_t);

        // 单个节点的最大原始数据大小
        size_t max_data_size = (emb_dim * sizeof(float)) + 
                            (loc_dim * sizeof(float)) +
                            sizeof(uint32_t) + // nnbr
                            (max_nbr_len * nbr_data_size);

        // 对齐后的节点块大小
        size_t max_aligned_size = disk::align_to_page_size(max_data_size);
        
        std::cout << "Node block size: " << max_aligned_size << " bytes" << std::endl;

        // 缓冲区复用
        std::vector<char> node_buffer(max_aligned_size);

        // --- 4. 循环读取节点数据 ---

        for (size_t i = 0; i < node_num; i++) {
            // 读取整个对齐后的块
            in.read(node_buffer.data(), max_aligned_size);
            
            char* current_ptr = node_buffer.data();

            // 4.1 读取 Embedding
            std::vector<float> emb(emb_dim);
            std::memcpy(emb.data(), current_ptr, emb_dim * sizeof(float));
            current_ptr += emb_dim * sizeof(float);

            // 4.2 读取 Location
            std::vector<float> loc(loc_dim);
            std::memcpy(loc.data(), current_ptr, loc_dim * sizeof(float));
            current_ptr += loc_dim * sizeof(float);

            // TODO: 将 emb 和 loc 设置到你的索引数据结构中
            // final_index_->setNodeData(i, emb, loc);

            // 4.3 读取邻居数量
            uint32_t nnbr = 0;
            std::memcpy(&nnbr, current_ptr, sizeof(uint32_t));
            current_ptr += sizeof(uint32_t);

            // 4.4 读取邻居列表
            // 注意：虽然缓冲区后面可能有 max_nbr_len 个空间，但有效数据只有 nnbr 个
            for (size_t j = 0; j < nnbr; j++) {
                uint32_t neighbor_id;
                std::memcpy(&neighbor_id, current_ptr, sizeof(uint32_t));
                current_ptr += sizeof(uint32_t);

                // 读取 alpha range (int8_t)
                std::vector<int8_t> alpha_range_int8(max_alpha_range_len * 2);
                std::memcpy(alpha_range_int8.data(), current_ptr, max_alpha_range_len * 2 * sizeof(int8_t));
                current_ptr += max_alpha_range_len * 2 * sizeof(int8_t);

                // 转换 int8_t -> float pair
                std::vector<std::pair<float, float>> available_range;
                for (size_t k = 0; k < max_alpha_range_len; k++) {
                    int8_t x = alpha_range_int8[2 * k];
                    int8_t y = alpha_range_int8[2 * k + 1];
                    
                    // 还原逻辑：如果在写入时填充了默认值(如0)，这里解析出来就是 0.0
                    // 通常需要判断是否是有效范围，但这取决于你的具体应用逻辑。
                    // 如果写入时是固定 resize 到 max_len，这里就读出所有的。
                    // 假如 0,0 代表无效，可以在这里过滤，或者全部存入。
                    
                    // 简单还原：
                    float range_start = static_cast<float>(x) / 100.0f;
                    float range_end = static_cast<float>(y) / 100.0f;
                    
                    // 只有当范围有意义时才添加 (假设非0即有效，或者根据具体业务逻辑调整)
                    if (x != 0 || y != 0) { 
                        available_range.emplace_back(range_start, range_end);
                    }
                }

                // TODO: 将邻居信息添加到图中
                // final_index_->addNeighbor(i, neighbor_id, available_range);
            }

            // 进度打印
            if (i % 10000 == 0) {
                std::cout << "Loaded " << i << " / " << node_num << " nodes." << std::endl;
            }
        }

        in.close();
        std::cout << "Graph loaded successfully." << std::endl;
        return;
    }
}