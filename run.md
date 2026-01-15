
```sh
./test/main deg openimage 0.5 1 1 disk
```

```sh
./test/search \
/home/gongwei/deg/dataset/Openimage/base_img_emb.fvecs \
/home/gongwei/deg/dataset/Openimage/base_text_emb.fvecs \
/home/gongwei/deg/saved_index/deg_openimage.index \
200 10 /home/gongwei/deg/dataset/alpha.bin \
/home/gongwei/deg/dataset/Openimage/query_img_emb.fvecs \
/home/gongwei/deg/dataset/Openimage/query_text_emb.fvecs \
/home/gongwei/deg/dataset/Openimage/gt/base_gt_round_00/top10_results.ivecs \
8 1 1
./test/build \
/home/gongwei/deg/dataset/Openimage/base_img_emb.fvecs \
/home/gongwei/deg/dataset/Openimage/base_text_emb.fvecs \
/home/gongwei/deg/saved_index/deg_openimage.index \
200 40 16 1 1
```


```sh
./convert_tool \
/home/gongwei/deg/dataset/Openimage/base_img_emb.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/base_img_emb.fvecs
./convert_tool \
/home/gongwei/deg/dataset/Openimage/base_text_emb.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/base_text_emb.fvecs 
./convert_tool \
/home/gongwei/deg/dataset/Openimage/query_img_emb.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/query_img_emb.fvecs
./convert_tool \
/home/gongwei/deg/dataset/Openimage/query_text_emb.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/query_text_emb.fvecs 
```
```sh
./test/disk_build  \
/home/gongwei/deg/dataset/Openimage_disk/base_img_emb.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/base_text_emb.fvecs \
/home/gongwei/deg/Index/openimage/ \
200 40 disk 24

./test/disk_build  \
/home/gongwei/deg/dataset/CC3M_disk/base_img_emb.fvecs \
/home/gongwei/deg/dataset/CC3M_disk/base_text_emb.fvecs \
/home/gongwei/deg/Index/CC3M/ \
200 40 disk 24

./test/disk_build  \
/home/gongwei/deg/dataset/Howto100M_disk/base_img_emb.fvecs \
/home/gongwei/deg/dataset/Howto100M_disk/base_text_emb.fvecs \
/home/gongwei/deg/Index/Howto100M/ \
200 40 disk 24
```

```sh
g++ -std=c++17 -O3 -fopenmp process_topk_alpha_file.cpp -o ptopk -lstdc++fs
./ptopk OpenImage/base_img_emb.fvecs OpenImage/base_text_emb.fvecs OpenImage/query_img_emb.fvecs OpenImage/query_text_emb.fvecs OpenImage/gt 1

./ptopk Openimage_disk/base_img_emb.fvecs \
Openimage_disk/base_text_emb.fvecs \
Openimage_disk/query_img_emb.fvecs \
Openimage_disk/query_text_emb.fvecs \
Openimage_disk/ alpha.bin

./ptopk Howto100M_disk/base_img_emb.fvecs \
Howto100M_disk/base_text_emb.fvecs \
Howto100M_disk/query_img_emb.fvecs \
Howto100M_disk/query_text_emb.fvecs \
Howto100M_disk/ alpha.bin

./ptopk CC3M_disk/base_img_emb.fvecs \
CC3M_disk/base_text_emb.fvecs \
CC3M_disk/query_img_emb.fvecs \
CC3M_disk/query_text_emb.fvecs \
CC3M_disk/ alpha.bin

./ptopk Openimage_disk/base_img_emb.fvecs \
Openimage_disk/base_text_emb.fvecs \
Openimage_disk/query_img_emb.fvecs \
Openimage_disk/query_text_emb.fvecs \
Openimage_disk/0.5/ alpha.bin

./ptopk Howto100M_disk/base_img_emb.fvecs \
Howto100M_disk/base_text_emb.fvecs \
Howto100M_disk/query_img_emb.fvecs \
Howto100M_disk/query_text_emb.fvecs \
Howto100M_disk/0.5/ alpha.bin

./ptopk CC3M_disk/base_img_emb.fvecs \
CC3M_disk/base_text_emb.fvecs \
CC3M_disk/query_img_emb.fvecs \
CC3M_disk/query_text_emb.fvecs \
CC3M_disk/0.5/ alpha.bin
```


```sh
GreatorPlus
mkdir build 
cd build
cmake ..
make -j

./tests/search_disk_index float \
/home/gongwei/deg/Index/openimage/ \
32 4 \
/home/gongwei/deg/dataset/Openimage_disk/query_img_emb.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/query_text_emb.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/base_gt_round_00/top10_results.ivecs \
10 l2 3 0 1 40

./tests/search_disk_index float \
/home/gongwei/deg/Index/Howto100M/ \
24 4 \
/home/gongwei/deg/dataset/Howto100M_disk/query_img_emb.fvecs \
/home/gongwei/deg/dataset/Howto100M_disk/query_text_emb.fvecs \
/home/gongwei/deg/dataset/alpha.fvecs \
/home/gongwei/deg/dataset/Howto100M_disk/top10_results.ivecs_ \
10 l2 3 0 1 100

./tests/search_disk_index float \
/home/gongwei/deg/Index/CC3M/ \
32 4 \
/home/gongwei/deg/dataset/CC3M_disk/query_img_emb.fvecs \
/home/gongwei/deg/dataset/CC3M_disk/query_text_emb.fvecs \
/home/gongwei/deg/dataset/alpha.fvecs \
/home/gongwei/deg/dataset/CC3M_disk/top10_results.ivecs_ \
10 l2 3 0 1 100

./tests/search_disk_index float \
/home/gongwei/deg/Index/openimage/ \
32 4 \
/home/gongwei/deg/dataset/Openimage_disk/query_img_emb.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/query_text_emb.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/base_gt_round_00/query_alpha.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/base_gt_round_00/top10_results.ivecs \
10 l2 3 0 1 40

./tests/search_disk_index float \
/home/gongwei/deg/Index/Howto100M/ \
24 4 \
/home/gongwei/deg/dataset/Howto100M_disk/query_img_emb.fvecs \
/home/gongwei/deg/dataset/Howto100M_disk/query_text_emb.fvecs \
/home/gongwei/deg/dataset/Howto100M_disk/base_gt_round_00/query_alpha.fvecs \
/home/gongwei/deg/dataset/Howto100M_disk/base_gt_round_00/top10_results.ivecs \
10 l2 3 0 1 100

./tests/search_disk_index float \
/home/gongwei/deg/Index/CC3M/ \
32 4 \
/home/gongwei/deg/dataset/CC3M_disk/query_img_emb.fvecs \
/home/gongwei/deg/dataset/CC3M_disk/query_text_emb.fvecs \
/home/gongwei/deg/dataset/CC3M_disk/base_gt_round_00/query_alpha.fvecs \
/home/gongwei/deg/dataset/CC3M_disk/base_gt_round_00/top10_results.ivecs \
10 l2 3 0 1 100
```

./tests/search_disk_index float \
/home/gongwei/deg/Index/openimage/ \
32 4 \
/home/gongwei/deg/dataset/Openimage_disk/query_img_emb.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/query_text_emb.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/base_gt_round_00/query_alpha.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/base_gt_round_00/top10_results.ivecs \
10 l2 3 0 1 10
./tests/search_disk_index float \
/home/gongwei/deg/Index/openimage/ \
32 4 \
/home/gongwei/deg/dataset/Openimage_disk/query_img_emb.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/query_text_emb.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/base_gt_round_00/query_alpha.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/base_gt_round_00/top10_results.ivecs \
10 l2 3 0 1 20
./tests/search_disk_index float \
/home/gongwei/deg/Index/openimage/ \
32 4 \
/home/gongwei/deg/dataset/Openimage_disk/query_img_emb.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/query_text_emb.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/base_gt_round_00/query_alpha.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/base_gt_round_00/top10_results.ivecs \
10 l2 3 0 1 30
./tests/search_disk_index float \
/home/gongwei/deg/Index/openimage/ \
32 4 \
/home/gongwei/deg/dataset/Openimage_disk/query_img_emb.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/query_text_emb.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/base_gt_round_00/query_alpha.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/base_gt_round_00/top10_results.ivecs \
10 l2 3 0 1 40
./tests/search_disk_index float \
/home/gongwei/deg/Index/openimage/ \
32 4 \
/home/gongwei/deg/dataset/Openimage_disk/query_img_emb.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/query_text_emb.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/base_gt_round_00/query_alpha.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/base_gt_round_00/top10_results.ivecs \
10 l2 3 0 1 50
./tests/search_disk_index float \
/home/gongwei/deg/Index/openimage/ \
32 4 \
/home/gongwei/deg/dataset/Openimage_disk/query_img_emb.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/query_text_emb.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/base_gt_round_00/query_alpha.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/base_gt_round_00/top10_results.ivecs \
10 l2 3 0 1 60
./tests/search_disk_index float \
/home/gongwei/deg/Index/openimage/ \
32 4 \
/home/gongwei/deg/dataset/Openimage_disk/query_img_emb.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/query_text_emb.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/base_gt_round_00/query_alpha.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/base_gt_round_00/top10_results.ivecs \
10 l2 3 0 1 70
./tests/search_disk_index float \
/home/gongwei/deg/Index/openimage/ \
32 4 \
/home/gongwei/deg/dataset/Openimage_disk/query_img_emb.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/query_text_emb.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/base_gt_round_00/query_alpha.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/base_gt_round_00/top10_results.ivecs \
10 l2 3 0 1 80
./tests/search_disk_index float \
/home/gongwei/deg/Index/openimage/ \
32 4 \
/home/gongwei/deg/dataset/Openimage_disk/query_img_emb.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/query_text_emb.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/base_gt_round_00/query_alpha.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/base_gt_round_00/top10_results.ivecs \
10 l2 3 0 1 90
./tests/search_disk_index float \
/home/gongwei/deg/Index/openimage/ \
32 4 \
/home/gongwei/deg/dataset/Openimage_disk/query_img_emb.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/query_text_emb.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/base_gt_round_00/query_alpha.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/base_gt_round_00/top10_results.ivecs \
10 l2 3 0 1 100
./tests/search_disk_index float \
/home/gongwei/deg/Index/openimage/ \
32 4 \
/home/gongwei/deg/dataset/Openimage_disk/query_img_emb.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/query_text_emb.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/base_gt_round_00/query_alpha.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/base_gt_round_00/top10_results.ivecs \
10 l2 3 0 1 150
./tests/search_disk_index float \
/home/gongwei/deg/Index/openimage/ \
32 4 \
/home/gongwei/deg/dataset/Openimage_disk/query_img_emb.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/query_text_emb.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/base_gt_round_00/query_alpha.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/base_gt_round_00/top10_results.ivecs \
10 l2 3 0 1 200
./tests/search_disk_index float \
/home/gongwei/deg/Index/openimage/ \
32 4 \
/home/gongwei/deg/dataset/Openimage_disk/query_img_emb.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/query_text_emb.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/base_gt_round_00/query_alpha.fvecs \
/home/gongwei/deg/dataset/Openimage_disk/base_gt_round_00/top10_results.ivecs \
10 l2 3 0 1 400



