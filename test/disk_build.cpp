#include "nbr/abstract_nbr.h"
#include "nbr/nbr.h"
#include "policy.h"
#include "utils.h"
#include <builder.h>
#include <memory>
#include <set_para.h>
#include <disk.h>
#include <iostream>

void DEG(stkq::Parameters &parameters)
{
    const unsigned num_threads = parameters.get<unsigned>("n_threads");
    std::string base_emb_path = parameters.get<std::string>("base_emb_path");
    std::string base_loc_path = parameters.get<std::string>("base_loc_path");
    std::string graph_file = parameters.get<std::string>("graph_file");
    std::string disk_index_file = parameters.get<std::string>("disk_index_file");
    std::string disk_index_path = parameters.get<std::string>("disk_index_path");
    std::string disk_meta = disk_index_path + "_disk.index";
    std::string disk_topo = disk_index_path + "disk_index_graph";
    std::string disk_data = disk_index_path + "disk_index_data";
    if (parameters.get<std::string>("exc_type") == "disk")
    {
        auto *builder = new stkq::IndexBuilder(num_threads, parameters.get<float>("max_emb_distance"), parameters.get<float>("max_spatial_distance"));
        pipeann::AbstractNeighbor<float> *nbr_emb_handler = pipeann::get_nbr_handler<float>(pipeann::Metric::L2, "pq"); 
        nbr_emb_handler->build(disk_index_path + "emb", base_emb_path, 256);
        pipeann::AbstractNeighbor<float> *nbr_loc_handler = pipeann::get_nbr_handler<float>(pipeann::Metric::L2, "pq"); 
        nbr_loc_handler->build(disk_index_path + "loc", base_loc_path, 256);

        // build disk indx
        builder->load(&base_emb_path[0], &base_loc_path[0], "", "", "", "", parameters);
        builder->peak_memory_footprint();
        builder->load_graph(stkq::TYPE::INDEX_DEG, &graph_file[0]);
        builder->peak_memory_footprint();
        builder->save_graph_disk(stkq::TYPE::INDEX_DEG, &disk_meta[0], &disk_topo[0], &disk_data[0]);
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
        std::cout << "./disk_build vec_base_emb vec_base_loc graph_index disk_index_path maximum_spatial_distance maximum_emb_distance exc_type"
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

    parameters.set<float>("max_spatial_distance", std::stof(maximum_spatial_distance));
    parameters.set<float>("max_emb_distance", std::stof(maximum_emb_distance));

    std::cout << "max_emb_distance: " << maximum_emb_distance << std::endl;
    std::cout << "max_spatial_distance: " << maximum_spatial_distance << std::endl;
    parameters.set<std::string>("exc_type", exc_type);

    DEG(parameters);
    return 0;
}