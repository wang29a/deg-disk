
#include "disk_util.h"
#include "index.h"
#include <cstddef>
#include <cstdint>
#include <libaio.h>
#include <fcntl.h>
#include <libaio.h>
#include <unistd.h>

namespace disk {
    class QueryData {
    public:
        void load(char *query_emb_file, char *query_loc_file, char *query_alpha_file, char *ground_file, stkq::Parameters &parameters);

        float *getQueryEmbData() const
        {
            return query_emb_data_;
        }

        void setQueryEmbData(float *queryEmbData)
        {
            query_emb_data_ = queryEmbData;
        }

        float *getQueryLocData() const
        {
            return query_loc_data_;
        }

        void setQueryLocData(float *queryLocData)
        {
            query_loc_data_ = queryLocData;
        }

        float *getQueryWeightData() const
        {
            return query_alpha_;
        }

        void setQueryWeightData(float *queryWeightData)
        {
            query_alpha_ = queryWeightData;
        }

        unsigned int *getGroundData() const
        {
            return ground_data_;
        }

        void setGroundData(unsigned int *groundData)
        {
            ground_data_ = groundData;
        }

        unsigned int getQueryLen() const
        {
            return query_len_;
        }

        void setQueryLen(unsigned int queryLen)
        {
            query_len_ = queryLen;
        }

        unsigned int getGroundLen() const
        {
            return ground_len_;
        }

        void setGroundLen(unsigned int groundLen)
        {
            ground_len_ = groundLen;
        }

        unsigned int getQueryEmbDim() const
        {
            return query_emb_dim_;
        }

        unsigned int getQueryLocDim() const
        {
            return query_loc_dim_;
        }

        void setQueryEmbDim(unsigned int queryEmbDim)
        {
            query_emb_dim_ = queryEmbDim;
        }

        void setQueryLocDim(unsigned int queryLocDim)
        {
            query_loc_dim_ = queryLocDim;
        }

        unsigned int getGroundDim() const
        {
            return ground_dim_;
        }

        void setGroundDim(unsigned int groundDim)
        {
            ground_dim_ = groundDim;
        }

    private:
        float *query_emb_data_, *query_loc_data_, *query_alpha_;
        unsigned *ground_data_;

        unsigned query_len_, ground_len_;
        unsigned query_emb_dim_, query_loc_dim_, ground_dim_;
    };
    class DiskIndex {
    public:
        
        ~DiskIndex() {
            delete e_dist_;
            delete s_dist_;
            int64_t ret;
            // check to make sure file_desc is closed
            ret = fcntl(this->file_desc_, F_GETFD);
            if (ret == -1)
            {
                if (errno != EBADF)
                {
                    std::cerr << "close() not called" << std::endl;
                    // close file desc
                    ret = ::close(this->file_desc_);
                    // error checks
                    if (ret == -1)
                    {
                        std::cerr << "close() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno)
                                << std::endl;
                    }
                }
            }
            aligned_free((void *)emb_scratch);
            aligned_free((void *)loc_scratch);
            aligned_free((void *)sector_scratch);
        }
        
        void init();

        DiskIndex *search();

        void peak_memory_footprint();
        
        void SearchAtLayer(unsigned qnode,
                            stkq::Index::VisitedList *visited_list,
                            std::priority_queue<stkq::Index::DEG_FurtherFirst> &result);
        void RouteInner(unsigned int query, std::vector<stkq::Index::Neighbor> &pool,
                        std::vector<unsigned int> &res);

        void open_file();

        void load_query_data(char *query_emb_file, char *query_loc_file, char *query_alpha_file, char *ground_file, stkq::Parameters &parameters) {
            query_data.load(query_emb_file, query_loc_file, query_alpha_file, ground_file, parameters);
        }

        void load_metadata(const char *index_file);
    private:
        stkq::E_Distance *get_E_Dist() const
        {
            return e_dist_;
        }

        stkq::E_Distance *get_S_Dist() const
        {
            return s_dist_;
        }
        // sector # on disk where node_id is present with in the graph part
        uint64_t get_node_sector(uint64_t node_id);

        // ptr to start of the node
        char *offset_to_node(char *sector_buf, uint64_t node_id);

        // returns region of `node_buf` containing [NNBRS][{NBR_ID(uint32_t) ALPHA RANGE()}]
        uint32_t *offset_to_node_nhood(char *node_buf);

        // returns region of `node_buf` containing [EMB(FLOAT)]
        float *offset_to_node_emb(char *node_buf);

        // returns region of `node_buf` containing [LOC(FLOAT)]
        float *offset_to_node_loc(char *node_buf);

        int8_t *offset_to_node_nhood_alpha(char* nhood_buf);

        void setup_sector_scratch();

        unsigned int getDistCount() const
        {
            return dist_count;
        }

        void resetDistCount()
        {
            dist_count = 0;
        }

        void addDistCount()
        {
            dist_count += 1;
        }

        unsigned int getHopCount() const
        {
            return hop_count;
        }

        void resetHopCount()
        {
            hop_count = 0;
        }

        void addHopCount()
        {
            hop_count += 1;
        }

    private:
        
        uint32_t _max_node_len = 0;
        uint32_t _max_nbr_len = 0;
        uint32_t _max_alpha_range_len = 0;
        uint32_t _nnodes_per_sector = 0; // 0 for multi-sector nodes, >0 for multi-node sectors
        uint32_t _max_degree = 0;

        uint32_t _num_points = 0;
        uint32_t emb_dim_ = 0, loc_dim_ = 0;
        uint64_t _disk_bytes_per_point = 0; // Number of bytes

        stkq::E_Distance *e_dist_;
        stkq::E_Distance *s_dist_;
        std::vector<uint32_t> enterpoint_set;
        QueryData query_data;

        float alpha_;
        unsigned search_l_, k_;
        int file_desc_;
        io_context_t ctx_;

        float *emb_scratch = nullptr; // MUST BE AT LEAST [sizeof(T) * data_dim]
        float *loc_scratch = nullptr; // MUST BE AT LEAST [sizeof(T) * data_dim]

        char *sector_scratch = nullptr; // MUST BE AT LEAST [MAX_N_SECTOR_READS * SECTOR_LEN]
        size_t sector_idx = 0;          // index of next [SECTOR_LEN] scratch to use

        unsigned dist_count = 0;
        unsigned hop_count = 0;

    };
    void execute_io(io_context_t ctx, int fd, std::vector<AlignedRead> &read_reqs, uint64_t n_retries = 0);
}