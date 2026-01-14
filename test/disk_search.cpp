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
    std::string query_emb_path = parameters.get<std::string>("query_emb_path");
    std::string query_loc_path = parameters.get<std::string>("query_loc_path");
    std::string query_alpha_path = parameters.get<std::string>("query_alpha_path");
    std::string ground_path = parameters.get<std::string>("ground_path");
    std::string disk_index_file = parameters.get<std::string>("disk_index_file");
    std::string disk_index_path = parameters.get<std::string>("disk_index_path");
    auto disk_index = std::make_shared<disk::DiskIndex>();
    disk_index->init();
    auto *builder = new stkq::IndexBuilder(num_threads, parameters.get<float>("max_emb_distance"), parameters.get<float>("max_spatial_distance"));
    if (parameters.get<std::string>("exc_type") == "search")
    {
        // search
        disk_index->load_query_data(&query_emb_path[0], &query_loc_path[0], &query_alpha_path[0], &ground_path[0], parameters);
        disk_index->load_pq_data(disk_index_path);
        builder->peak_memory_footprint();
        disk_index->load_metadata(disk_index_file.data());
        builder->peak_memory_footprint();
        disk_index->search();
        // builder->search_disk(stkq::TYPE::SEARCH_ENTRY_NONE, stkq::TYPE::ROUTER_DEG, stkq::TYPE::L_SEARCH_ASCEND, parameters);
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

    if (argc != 7)
    {
        std::cout << "./main vec_base_emb vec_base_loc disk_index_path maximum_spatial_distance maximum_emb_distance exc_type"
                  << std::endl;
        exit(-1);
    }
    stkq::Parameters parameters;
    parameters.set<std::string>("base_emb_path", argv[1]);
    parameters.set<std::string>("base_loc_path", argv[2]);
    parameters.set<std::string>("graph_file", argv[3]);
    parameters.set<std::string>("disk_index_path", argv[4]);
    std::string disk_index_file = argv[4];
    disk_index_file += "disk.index";
    parameters.set<std::string>("disk_index_file", disk_index_file);
    parameters.set<unsigned>("n_threads", 8);

    std::string maximum_spatial_distance(argv[5]);
    std::string maximum_emb_distance(argv[6]);
    std::string exc_type(argv[7]);

    parameters.set<std::string>("base_emb", argv[1]);
    parameters.set<std::string>("base_loc", argv[2]);
    parameters.set<std::string>("graph_index", argv[3]);
    std::string dataset_root = R"(/home/gongwei/deg/dataset/)";
    std::string index_path = R"(/home/gongwei/deg/saved_index/)";
    parameters.set<std::string>("dataset_root", dataset_root);
    parameters.set<std::string>("index_path", index_path);
    parameters.set<unsigned>("n_threads", 8);

    std::string maximum_spatial_distance(argv[4]);
    std::string maximum_emb_distance(argv[5]);
    std::string exc_type(argv[6]);

    parameters.set<float>("max_spatial_distance", std::stof(maximum_spatial_distance));
    parameters.set<float>("max_emb_distance", std::stof(maximum_emb_distance));

    std::cout << "max_emb_distance: " << maximum_emb_distance << std::endl;
    std::cout << "max_spatial_distance: " << maximum_spatial_distance << std::endl;
    parameters.set<std::string>("exc_type", exc_type);
    DISK(parameters);
    return 0;
}