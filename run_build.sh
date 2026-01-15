#!/bin/bash

# =================================================================
# 默认参数设置 (Default Configuration)
# 根据你提供的命令设置默认值
# =================================================================
BASE_IMG="/home/gongwei/deg/dataset/Openimage_disk/base_img_emb.fvecs"
BASE_TEXT="/home/gongwei/deg/dataset/Openimage_disk/base_text_emb.fvecs"
OUTPUT_INDEX_PATH="/home/gongwei/deg/Index/openimage/"

L_BUILD=200      # 构建时的队列长度 (Queue Length)
DEGREE=40        # 邻居数 (Neighbors / Degree)
THREADS=24       # 线程数 (Threads)

# 固定参数
MODE="disk"      # 模式，原命令中为 disk

# =================================================================
# 帮助函数 (Help Function)
# =================================================================
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -m <path>    Base Image path (图像数据路径)"
    echo "  -t <path>    Base Text path (文本数据路径)"
    echo "  -o <path>    Output Index path (索引输出路径)"
    echo "  -l <num>     Build queue length (构建队列长度, default: 200)"
    echo "  -d <num>     Number of neighbors/degree (邻居数, default: 40)"
    echo "  -n <num>     Number of threads (线程数, default: 24)"
    echo "  -h           Show this help message"
    exit 1
}

# =================================================================
# 参数解析 (Argument Parsing)
# =================================================================
while getopts "m:t:o:l:d:n:h" opt; do
  case $opt in
    m) BASE_IMG="$OPTARG" ;;
    t) BASE_TEXT="$OPTARG" ;;
    o) OUTPUT_INDEX_PATH="$OPTARG" ;;
    l) L_BUILD="$OPTARG" ;;
    d) DEGREE="$OPTARG" ;;
    n) THREADS="$OPTARG" ;;
    h) usage ;;
    *) usage ;;
  esac
done

# =================================================================
# 执行命令 (Execution)
# =================================================================

# 可执行文件路径
EXE_PATH="./build/test/disk_build"

if [ ! -f "$EXE_PATH" ]; then
    echo "Error: Executable $EXE_PATH not found!"
    # 尝试检查是否在当前目录下 (容错处理)
    if [ -f "./disk_build" ]; then
        EXE_PATH="./disk_build"
    else
        exit 1
    fi
fi

echo "------------------------------------------------"
echo "Starting Disk Build with parameters:"
echo "Output Path:  $OUTPUT_INDEX_PATH"
echo "Queue Length: $L_BUILD"
echo "Neighbors:    $DEGREE"
echo "Threads:      $THREADS"
echo "------------------------------------------------"

# 创建输出目录（如果不存在）
if [ ! -d "$OUTPUT_INDEX_PATH" ]; then
    echo "Creating output directory: $OUTPUT_INDEX_PATH"
    mkdir -p "$OUTPUT_INDEX_PATH"
fi

# 运行命令
set -x # 开启调试模式，显示具体执行的命令
$EXE_PATH \
    "$BASE_IMG" \
    "$BASE_TEXT" \
    "$OUTPUT_INDEX_PATH" \
    "$L_BUILD" "$DEGREE" "$MODE" "$THREADS"
set +x # 关闭调试模式
