
#include "defaults.h"
#include "disk.h"
#include "disk_util.h"
#include "index.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ostream>
#include <queue>
#include <vector>

namespace disk {
    inline uint64_t DiskIndex::get_node_sector(uint64_t node_id) 
    {
        return 1 + (_nnodes_per_sector > 0 ? node_id / _nnodes_per_sector
                                        : node_id * DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN));
    }

    inline char *DiskIndex::offset_to_node(char *sector_buf, uint64_t node_id)
    {
        return sector_buf + (_nnodes_per_sector == 0 ? 0 : (node_id % _nnodes_per_sector) * _max_node_len);
    }

    inline uint32_t *DiskIndex::offset_to_node_nhood(char *node_buf)
    {
        return (uint32_t *)(node_buf + emb_dim_*sizeof(float) + loc_dim_*sizeof(float));
    }

    inline float *DiskIndex::offset_to_node_emb(char *node_buf)
    {
        return (float *)(node_buf);
    }

    inline float *DiskIndex::offset_to_node_loc(char *node_buf)
    {
        return (float *)(node_buf + emb_dim_*sizeof(float));
    }

    inline int8_t *DiskIndex::offset_to_node_nhood_alpha(char* nhood_buf) {
        return (int8_t *)(nhood_buf+sizeof(uint32_t));
    }

    void DiskIndex::setup_sector_scratch() {
        size_t emb_alloc_size = ROUND_UP(sizeof(float) * emb_dim_, 256);
        size_t loc_alloc_size = ROUND_UP(sizeof(float) * loc_dim_, 256);

        std::cout<< emb_alloc_size << " " << loc_alloc_size << std::endl;
        alloc_aligned((void **)&emb_scratch, emb_alloc_size, 256);
        alloc_aligned((void **)&loc_scratch, loc_alloc_size, 256);
        alloc_aligned((void **)&sector_scratch, defaults::MAX_N_SECTOR_READS * defaults::SECTOR_LEN,
                   defaults::SECTOR_LEN);
        // ::alloc_aligned((void **)&this->_aligned_query_T, aligned_dim * sizeof(T), 8 * sizeof(T));

        memset(emb_scratch, 0, emb_alloc_size);
        memset(loc_scratch, 0, loc_alloc_size);

    }

    DiskIndex *DiskIndex::search()
    {
        std::cout << "__SEARCH__" << std::endl;
        unsigned K = 10; // 在近邻搜索中要找到的最近邻的数量
        std::vector<std::vector<unsigned>> res;
        std::set<unsigned> visited;
        unsigned L = 0;
        visited.insert(L);
        for (unsigned t = 0; t < 20; t++)
        {

            L = L + K;
            std::cout << "SEARCH_L : " << L << std::endl;
            if (L < K) {
                std::cout << "search_L cannot be smaller than search_K! " << std::endl;
                exit(-1);
            }

            auto s1 = std::chrono::high_resolution_clock::now();
            search_l_ = L;

            res.clear();
            res.resize(query_data.getQueryLen());
            //  #pragma omp parallel for
            for (unsigned i = 0; i < query_data.getQueryLen(); i++)
            //                for (unsigned i = 0; i < 1000; i++)
            {
                alpha_ = query_data.getQueryWeightData()[i];
                std::vector<stkq::Index::Neighbor> pool;
                RouteInner(i, pool, res[i]);
            }
            auto e1 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = e1 - s1;
            std::cout << "search time: " << diff.count() / query_data.getQueryLen() << std::endl;
            std::cout << "DistCount: " << getDistCount() << std::endl;
            std::cout << "HopCount: " << getHopCount() << std::endl;
            resetDistCount();
            resetHopCount();
            std::cout << "qps: " << query_data.getQueryLen() / diff.count() << std::endl;
            int cnt = 0;
            float recall = 0;
            for (unsigned i = 0; i < query_data.getQueryLen(); i++)
            {
                if (res[i].size() == 0)
                    continue;
                float tmp_recall = 0;
                float cnt = 0;
                for (unsigned j = 0; j < K; j++)
                {
                    unsigned k = 0;
                    for (; k < K; k++)
                    {
                        if (res[i][j] == query_data.getGroundData()[i * query_data.getGroundDim() + k])
                            break;
                    }
                    if (k == K)
                        cnt++;
                }
                tmp_recall = (float)(K - cnt) / (float)K;
                recall = recall + tmp_recall;
            }
            float acc = recall / query_data.getQueryLen();
            std::cout << K << " NN accuracy: " << acc << std::endl;
        }
        // e = std::chrono::high_resolution_clock::now();
        std::cout << "__SEARCH FINISH__" << std::endl;

        return this;
    }

    void DiskIndex::RouteInner(unsigned int query, std::vector<stkq::Index::Neighbor> &pool,
                                std::vector<unsigned int> &res)
    {
        /*
            需要load的请求
            load
            加入优先队列、处理邻居
        */
        auto *visited_list = new stkq::Index::VisitedList(_num_points);
        visited_list->Reset();
        unsigned visited_mark = visited_list->GetVisitMark();
        unsigned int *visited = visited_list->GetVisited();

        std::priority_queue<stkq::Index::DEG_FurtherFirst> result;
        std::priority_queue<stkq::Index::DEG_CloserFirst> tmp;

        SearchAtLayer(query, visited_list, result);

        // std::cout<< "result size: " << result.size() << std::endl;

        while (!result.empty())
        {
            tmp.push(stkq::Index::DEG_CloserFirst(result.top().GetId(), result.top().GetDistance()));
            result.pop();
        }

        res.resize(k_);
        int pos = 0;
        while (!tmp.empty() && pos < k_)
        {
            auto top_node = tmp.top().GetId();
            tmp.pop();
            res[pos] = top_node;
            pos++;
        }

        delete visited_list;
    }

    void DiskIndex::SearchAtLayer(unsigned qnode,
                                    stkq::Index::VisitedList *visited_list,
                                    std::priority_queue<stkq::Index::DEG_FurtherFirst> &result)
    {
        const auto L = search_l_;

        std::priority_queue<stkq::Index::DEG_CloserFirst> candidates;
        visited_list->Reset();

        bool m_first = false;

        std::vector<unsigned> load;
        std::vector<std::pair<unsigned, char*>> load_datas;
        std::vector<AlignedRead> read_reqs;
        sector_idx = 0;
        const uint64_t num_sectors_per_node =
            _nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);

        for (size_t i = 0; i < this->enterpoint_set.size(); i++)
        {
            auto id = this->enterpoint_set[i];
            load.emplace_back(id);
        }
        for (size_t i = 0; i < load.size(); i++)
        {
            auto id = load[i];
            std::pair<unsigned, char*> load_data;
            load_data.first = id;
            load_data.second = sector_scratch + num_sectors_per_node * sector_idx * defaults::SECTOR_LEN;
            sector_idx ++;

            load_datas.emplace_back(load_data);
            // std::cout<< "ep: " << id << " offest: " << get_node_sector(id)*defaults::SECTOR_LEN << std::endl;
            read_reqs.emplace_back(get_node_sector(id)*defaults::SECTOR_LEN,
                                   num_sectors_per_node*defaults::SECTOR_LEN, load_data.second);
        }

        execute_io(ctx_, file_desc_, read_reqs);
        for (auto &data: load_datas) {
            auto id = data.first;
            char *node_disk_buf = offset_to_node(data.second, data.first);
            uint32_t *node_buf = offset_to_node_nhood(node_disk_buf);
            uint64_t nnbrs = (uint64_t)(*node_buf);
            float *node_fp_emb = offset_to_node_emb(data.second);
            float *node_fp_loc = offset_to_node_loc(data.second);
            memcpy(emb_scratch, node_fp_emb, emb_dim_*sizeof(float));
            memcpy(loc_scratch, node_fp_loc, loc_dim_*sizeof(float));
            float cur_e_d =
                get_E_Dist()->
                    compare(query_data.getQueryEmbData() + (size_t)qnode*emb_dim_, emb_scratch, emb_dim_);
            addDistCount();

            float cur_s_d = 
                get_S_Dist()->
                    compare(query_data.getQueryLocData() + (size_t)qnode*loc_dim_, loc_scratch, loc_dim_);
            addDistCount();

            float cur_dist = alpha_ * cur_e_d + (1 - alpha_) * cur_s_d;

            // std::cout<< "id: " << id << " dist: " << cur_dist << " " << cur_e_d << " " << cur_s_d << " " << alpha_ << std::endl;

            result.emplace(id, cur_dist);
            candidates.emplace(id, cur_dist);

            visited_list->MarkAsVisited(id);
        }
        auto top1 = candidates.top();

        // while (!candidates.empty()) {
        //     candidates.pop();
        // }
        // candidates.push(top1);

        while (!candidates.empty())
        {
            load.clear();
            load_datas.clear();
            read_reqs.clear();
            sector_idx = 0;
            const stkq::Index::DEG_CloserFirst &candidate = candidates.top();
            float lower_bound = result.top().GetDistance();
            if (candidate.GetDistance() > lower_bound) {
                break;
            }

            auto candidate_id = candidate.GetId();
            candidates.pop();
            addHopCount();
            load.emplace_back(candidate_id);

            // 加载数据 目标是邻居
            if (!load.empty()) {
                for (size_t i = 0; i < load.size(); i ++) {
                    auto id = load[i];
                    std::pair<unsigned, char*> load_data;
                    load_data.first = id;
                    load_data.second = sector_scratch + num_sectors_per_node * sector_idx * defaults::SECTOR_LEN;
                    sector_idx ++;

                    load_datas.emplace_back(load_data);
                    read_reqs.emplace_back(get_node_sector(id)*defaults::SECTOR_LEN,
                                            num_sectors_per_node*defaults::SECTOR_LEN, load_data.second);
                }

                execute_io(ctx_, file_desc_, read_reqs);
            }

            load.clear();


            for (auto &data : load_datas) {
                auto id = data.first;
                char *node_disk_buf = offset_to_node(data.second, data.first);
                uint32_t *node_buf = offset_to_node_nhood(node_disk_buf);
                uint32_t nnbrs = (uint32_t)(*node_buf);
                
                visited_list->MarkAsVisited(id);

                // neighbor
                char *node_nbrs = (char *)(node_buf + 1);


                // TODO add to class
                uint32_t nbr_data_len = _max_alpha_range_len*2+sizeof(uint32_t);
                for (size_t i = 0; i < nnbrs; i ++) {
                    uint32_t neighbor_id = *(uint32_t*)(node_nbrs+i*nbr_data_len);
                    // assert(_max_nbr_len == (_max_alpha_range_len+sizeof(unsigned)));
                    std::vector<std::pair<int8_t, int8_t>> use_range;
                    for (size_t k = 0; k < _max_alpha_range_len; k ++) {
                        int8_t alpha1 = (*(int8_t *)(node_nbrs+nbr_data_len*i+sizeof(uint32_t)+k*2));
                        int8_t alpha2 = (*(int8_t *)(node_nbrs+nbr_data_len*i+sizeof(uint32_t)+k*2+1));
                        if(!(alpha1<=100) || !(alpha1>=0) || !(alpha2<=100) || !(alpha2>=0)){
                            break;
                        }
                        if(alpha1 == 0 && alpha2 == 0) {
                            break;
                        }
                        use_range.emplace_back(alpha1, alpha2);
                    }

                    bool search_flag = false;
                    for (int i = 0; i < use_range.size(); i++)
                    {
                        if (alpha_ * 100 >= use_range[i].first && alpha_ * 100 <= use_range[i].second) {
                            search_flag = true;
                            break;
                        }
                        if (alpha_ * 100 < use_range[i].first) {
                            break;
                        }
                        if (alpha_ * 100 > use_range[i].second) {
                            continue;
                        }
                    }

                    // search_flag = true;
                    if (search_flag) {
                        if (visited_list->NotVisited(neighbor_id)) {
                            visited_list->MarkAsVisited(neighbor_id);
                            load.emplace_back(neighbor_id);
                        }
                    }
                }
            }

            load_datas.clear();
            read_reqs.clear();
            sector_idx = 0;

            // 加载数据 目标是数据
            if (!load.empty()) {
                for (size_t i = 0; i < load.size(); i ++) {
                    auto id = load[i];
                    std::pair<unsigned, char*> load_data;
                    load_data.first = id;
                    load_data.second = sector_scratch + num_sectors_per_node * sector_idx * defaults::SECTOR_LEN;
                    sector_idx ++;

                    load_datas.emplace_back(load_data);
                    read_reqs.emplace_back(get_node_sector(id)*defaults::SECTOR_LEN,
                                            num_sectors_per_node*defaults::SECTOR_LEN, load_data.second);
                }

                execute_io(ctx_, file_desc_, read_reqs);
            }
            for (auto &data : load_datas) {
                auto id = data.first;
                // char *node_disk_buf = offset_to_node(data.second, data.first);
                // uint32_t *node_buf = offset_to_node_nhood(node_disk_buf);
                // uint32_t nnbrs = (uint32_t)(*node_buf);
                float *node_fp_emb = offset_to_node_emb(data.second);
                float *node_fp_loc = offset_to_node_loc(data.second);

                memcpy(emb_scratch, node_fp_emb, emb_dim_*sizeof(float));
                memcpy(loc_scratch, node_fp_loc, loc_dim_*sizeof(float));
                

                if (result.size() >= L) {
                    if (m_first) {
                        float threshold = result.top().GetDistance();

                        float s_d = 
                            get_S_Dist()->
                                compare(query_data.getQueryLocData() + (size_t)qnode*loc_dim_,loc_scratch, loc_dim_);

            addDistCount();
                        if ((1 - alpha_) * s_d >= threshold)
                        {
                            continue;
                        }

                        float e_d =
                            get_E_Dist()->
                                compare(query_data.getQueryEmbData() + (size_t)qnode*emb_dim_, emb_scratch, emb_dim_);

            addDistCount();
                        float d = alpha_ * e_d + (1 - alpha_) * s_d;

                        if (threshold > d)
                        {
                            result.emplace(id, d);
                            candidates.emplace(id, d);
                            if (result.size() > L)
                                result.pop();
                        }
                    }
                    else
                    {
                        float threshold = result.top().GetDistance();

                        if (alpha_ <= 0.5)
                        {
                            float s_d = 
                                get_S_Dist()->
                                    compare(query_data.getQueryLocData() + (size_t)qnode*loc_dim_, loc_scratch, loc_dim_);

            addDistCount();
                            if ((1 - alpha_) * s_d >= threshold)
                            {
                                continue;
                            }

            addDistCount();
                            float e_d =
                                get_E_Dist()->
                                    compare(query_data.getQueryEmbData() + (size_t)qnode*emb_dim_, emb_scratch, emb_dim_);
                            float d = alpha_ * e_d + (1 - alpha_) * s_d;

                            if (threshold > d)
                            {
                                result.emplace(id, d);
                                candidates.emplace(id, d);
                                if (result.size() > L)
                                    result.pop();
                            }
                        }
                        else
                        {
                            float e_d =
                                get_E_Dist()->
                                    compare(query_data.getQueryEmbData() + (size_t)qnode*emb_dim_, emb_scratch, emb_dim_);

            addDistCount();
                            if (alpha_ * e_d >= threshold)
                            {
                                continue;
                            }

                            float s_d = 
                                get_S_Dist()->
                                    compare(query_data.getQueryLocData() + (size_t)qnode*loc_dim_,loc_scratch, loc_dim_);

            addDistCount();
                            float d = alpha_ * e_d + (1 - alpha_) * s_d;

                            if (threshold > d)
                            {
                                result.emplace(id, d);
                                candidates.emplace(id, d);
                                if (result.size() > L)
                                    result.pop();
                            }
                        }
                    }
                }
                else
                {
                    float e_d =
                        get_E_Dist()->
                            compare(query_data.getQueryEmbData() + (size_t)qnode*emb_dim_, emb_scratch, emb_dim_);

            addDistCount();
                    float s_d = 
                        get_S_Dist()->
                            compare(query_data.getQueryLocData() + (size_t)qnode*loc_dim_,loc_scratch, loc_dim_);

            addDistCount();
                    float d = alpha_ * e_d + (1 - alpha_) * s_d;
                    result.emplace(id, d);
                    candidates.emplace(id, d);
                    if (result.size() > L)
                        result.pop();
                }
            }


        }
    }
}