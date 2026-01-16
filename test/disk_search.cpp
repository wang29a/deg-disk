#include "nbr/abstract_nbr.h"
#include "nbr/nbr.h"
#include "policy.h"
#include "utils.h"
#include <builder.h>
#include <memory>
#include <set_para.h>
#include <disk.h>
#include <iostream>

void DISK(stkq::Parameters &parameters)
{
    const unsigned num_threads = parameters.get<unsigned>("n_threads");
    std::string query_emb_path = parameters.get<std::string>("query_emb");
    std::string query_loc_path = parameters.get<std::string>("query_loc");
    std::string query_alpha_path = parameters.get<std::string>("query_alpha");
    std::string ground_path = parameters.get<std::string>("query_gt");
    std::string disk_index_file = parameters.get<std::string>("disk_index_file");
    std::string disk_index_path = parameters.get<std::string>("disk_index_path");
    auto disk_index = std::make_shared<disk::DiskIndex>();
    disk_index->init();
    auto *builder = new stkq::IndexBuilder(num_threads, parameters.get<float>("max_emb_distance"), parameters.get<float>("max_spatial_distance"));
    if (parameters.get<std::string>("exc_type") == "search")
    {
        // search
        disk_index->load_query_data(&query_emb_path[0], &query_loc_path[0], &query_alpha_path[0], &ground_path[0], parameters);
        builder->peak_memory_footprint();
        disk_index->load_metadata(disk_index_file.data());
        builder->peak_memory_footprint();
        disk_index->search();
        builder->peak_memory_footprint();
    }
    else
    {
        std::cout << "exc_type input error!" << std::endl;
    }
}

int main(int argc, char **argv)
{
    // ./test/main baseline1 openimage 0.5 1 1 build
    // ./test/main baseline2 openimage 0.5 1 1 build
    // ./test/main deg openimage 0.5 1 1 build

    if (argc != 8)
    {
        std::cout << "./disk_search disk_index_path vec_query_emb vec_query_loc query_alpha query_gt thread K L"
                  << std::endl;
        exit(-1);
    }
    stkq::Parameters parameters;
    uint32_t arg_idx = 0;
    arg_idx ++;
    parameters.set<std::string>("disk_index_path", argv[arg_idx ++]);
    std::string disk_index_file = argv[4];
    disk_index_file += "baseline_disk.index";
    parameters.set<std::string>("disk_index_file", disk_index_file);
    parameters.set<unsigned>("n_threads", 8);

    parameters.set<float>("max_spatial_distance", 1);
    parameters.set<float>("max_emb_distance", 1);

    parameters.set<std::string>("query_emb", argv[arg_idx ++]);
    parameters.set<std::string>("query_loc", argv[arg_idx ++]);
    parameters.set<std::string>("query_alpha", argv[arg_idx ++]);
    parameters.set<std::string>("query_gt", argv[arg_idx ++]);

    unsigned int threads = static_cast<unsigned int>(std::stoul(argv[arg_idx ++]));
    parameters.set<unsigned>("n_threads", threads);
    unsigned int L = static_cast<unsigned int>(std::stoul(argv[arg_idx ++]));
    unsigned int K = static_cast<unsigned int>(std::stoul(argv[arg_idx ++]));
    parameters.set<unsigned>("L", L);
    parameters.set<unsigned>("K", K);

    parameters.set<unsigned>("max_m", 0);
    std::cout << ", L: " << parameters.get<unsigned>("L")
                << ", K: " << parameters.get<unsigned>("K")
                << ", query_alpha: " << parameters.get<std::string>("query_alpha")
                << ", query_emb: " << parameters.get<std::string>("query_emb")
                << ", query_loc: " << parameters.get<std::string>("query_loc")
                << ", query_gt: " << parameters.get<std::string>("query_gt")
                << ", threads: " << parameters.get<unsigned>("n_threads");
    parameters.set<std::string>("exc_type", "search");
    DISK(parameters);
    return 0;
}