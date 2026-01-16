#!/bin/bash

# =================================================================
# 默认参数设置 (Default Configuration)
# 基于你提供的命令行参数设定默认值
# =================================================================
INDEX_PATH="/data/Index/Openimage/"
QUERY_IMG="/data/dataset/Openimage_disk/query_img_emb.fvecs"
QUERY_TEXT="/data/dataset/Openimage_disk/query_text_emb.fvecs"
QUERY_ALPHA="/data/dataset/Openimage_disk/base_gt_round_00/query_alpha.fvecs"
RESULT_PATH="/data/dataset/Openimage_disk/base_gt_round_00/top10_results.ivecs"

THREADS=24
K=10
L=100

# =================================================================
# 帮助函数 (Help Function)
# =================================================================
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -i <path>    Index path (索引路径)"
    echo "  -m <path>    Query Image path (图像查询路径)"
    echo "  -t <path>    Query Text path (文本查询路径)"
    echo "  -a <path>    Query Alpha path (Alpha参数路径)"
    echo "  -r <path>    Result output path (结果输出路径)"
    echo "  -n <num>     Number of threads (线程数, default: 24)"
    echo "  -k <num>     Top K (default: 10)"
    echo "  -l <num>     Search queue length (L参数, default: 100)"
    echo "  -h           Show this help message"
    exit 1
}

# =================================================================
# 参数解析 (Argument Parsing)
# =================================================================
while getopts "i:m:t:a:r:n:k:l:h" opt; do
  case $opt in
    i) INDEX_PATH="$OPTARG" ;;
    m) QUERY_IMG="$OPTARG" ;;
    t) QUERY_TEXT="$OPTARG" ;;
    a) QUERY_ALPHA="$OPTARG" ;;
    r) RESULT_PATH="$OPTARG" ;;
    n) THREADS="$OPTARG" ;;
    k) K="$OPTARG" ;;
    l) L="$OPTARG" ;;
    h) usage ;;
    *) usage ;;
  esac
done

# =================================================================
# 执行命令 (Execution)
# =================================================================

# 可执行文件路径
EXE_PATH="./build/test/disk_search"

# 检查可执行文件是否存在
if [ ! -f "$EXE_PATH" ]; then
    echo "Error: Executable $EXE_PATH not found!"
    # 容错：尝试在当前目录查找
    if [ -f "./disk_search" ]; then
        EXE_PATH="./disk_search"
    else
        exit 1
    fi
fi

echo "------------------------------------------------"
echo "Starting Disk Search..."
echo "Index:   $INDEX_PATH"
echo "Threads: $THREADS | K: $K | L: $L"
echo "Output:  $RESULT_PATH"
echo "------------------------------------------------"

# 确保输出目录存在
OUT_DIR=$(dirname "$RESULT_PATH")
if [ ! -d "$OUT_DIR" ]; then
    echo "Creating directory: $OUT_DIR"
    mkdir -p "$OUT_DIR"
fi

# 运行命令
set -x # 开启调试模式，打印执行的命令
$EXE_PATH \
    "$INDEX_PATH" \
    "$QUERY_IMG" \
    "$QUERY_TEXT" \
    "$QUERY_ALPHA" \
    "$RESULT_PATH" \
    "$THREADS" "$K" "$L"
set +x # 关闭调试模式