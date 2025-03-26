#!/bin/bash

# 设置测试参数
MATRIX_SIZES=(100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000)
ITERATIONS=10
TARGET_ARCH="arm64"
HOST_ARCH=$(uname -m)
OUTPUT_DIR="results_row_major_${TARGET_ARCH}"

# ARM交叉编译器
ARM_CROSS_COMPILER="aarch64-linux-gnu-g++"
QEMU_USER="qemu-aarch64"

# 检查依赖是否安装
check_dependencies() {
    echo "检查必要依赖..."

    # 检查交叉编译器
    if ! command -v $ARM_CROSS_COMPILER &> /dev/null; then
        echo "错误：ARM交叉编译器未安装。请运行以下命令安装："
        echo "sudo apt update && sudo apt install g++-aarch64-linux-gnu"
        exit 1
    fi

    # 检查QEMU用户模式模拟器
    if ! command -v $QEMU_USER &> /dev/null; then
        echo "错误：QEMU用户模式模拟器未安装。请运行以下命令安装："
        echo "sudo apt update && sudo apt install qemu-user qemu-user-static"
        exit 1
    fi

    echo "所有依赖已安装"
}

# 创建C++测试文件
create_cpp_file() {
    echo "创建矩阵测试C++文件..."
    cat > row_major_benchmark.cpp << 'EOF'
#include <iostream>
#include <ctime>
#include <iomanip>
#include <cstring>
#include <cmath>
using namespace std;

// 初始化矩阵
void initializeMatrix(double** A, double** B, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = i + j;
            B[i][j] = i - j;
        }
    }
}

// **逐行访问** 的 Cache 优化算法
void rowMajorSum(double** A, double** B, double* sum, int n) {
    for (int i = 0; i < n; i++) {
        sum[i] = 0.0;
    }
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            sum[i] += B[j][i] * A[j][i]; // 逐行访问，提高 Cache 命中率
        }
    }
}

// 逐行访问 + 循环展开
void rowMajorSumUnrolled(double** A, double** B, double* sum, int n) {
    for (int i = 0; i < n; i++) {
        sum[i] = 0.0;
    }

    for (int j = 0; j < n; j++) {
        int i = 0;
        for (; i < n-3; i += 4) {
            sum[i] += B[j][i] * A[j][i];
            sum[i+1] += B[j][i+1] * A[j][i+1];
            sum[i+2] += B[j][i+2] * A[j][i+2];
            sum[i+3] += B[j][i+3] * A[j][i+3];
        }
        // 处理剩余元素
        for (; i < n; i++) {
            sum[i] += B[j][i] * A[j][i];
        }
    }
}

// 分块优化版本
void blockOptimizedSum(double** A, double** B, double* sum, int n) {
    const int blockSize = 32; // ARM架构缓存行大小通常为32-64字节

    for (int i = 0; i < n; i++) {
        sum[i] = 0.0;
    }

    for (int jj = 0; jj < n; jj += blockSize) {
        for (int ii = 0; ii < n; ii += blockSize) {
            // 处理当前块
            for (int j = jj; j < min(jj + blockSize, n); j++) {
                for (int i = ii; i < min(ii + blockSize, n); i++) {
                    sum[i] += B[j][i] * A[j][i];
                }
            }
        }
    }
}

// 运行时间测量
double measureTime(void (*func)(double**, double**, double*, int), double** A, double** B, double* sum, int n, int repeat) {
    struct timespec start, end;
    double total_time = 0.0;

    for (int r = 0; r < repeat; r++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        func(A, B, sum, n);
        clock_gettime(CLOCK_MONOTONIC, &end);

        double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        total_time += time_taken;
    }

    return total_time / repeat;  // 返回平均时间
}

// 验证结果正确性
bool verifyResults(double* sum1, double* sum2, int n) {
    const double epsilon = 1e-6;
    for (int i = 0; i < n; i++) {
        if (fabs(sum1[i] - sum2[i]) > epsilon) {
            cout << "结果不匹配: sum1[" << i << "]=" << sum1[i] << ", sum2[" << i << "]=" << sum2[i] << endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    // 参数检查
    if (argc < 2) {
        cerr << "用法: " << argv[0] << " <矩阵大小> [重复次数]" << endl;
        return 1;
    }

    int n = atoi(argv[1]);  // 矩阵大小
    int repeat = (argc >= 3) ? atoi(argv[2]) : 10;  // 重复次数，默认10

    cout << "矩阵大小: " << n << " x " << n << endl;
    cout << "重复次数: " << repeat << endl;

    // 分配动态内存
    double** A = new double*[n];
    double** B = new double*[n];
    for (int i = 0; i < n; i++) {
        A[i] = new double[n];
        B[i] = new double[n];
    }

    double* sum_basic = new double[n];
    double* sum_unroll = new double[n];
    double* sum_block = new double[n];

    // 初始化数据
    initializeMatrix(A, B, n);

    // 测量性能
    double time_basic = measureTime(rowMajorSum, A, B, sum_basic, n, repeat);
    double time_unroll = measureTime(rowMajorSumUnrolled, A, B, sum_unroll, n, repeat);
    double time_block = measureTime(blockOptimizedSum, A, B, sum_block, n, repeat);

    // 验证结果一致性
    bool basic_vs_unroll = verifyResults(sum_basic, sum_unroll, n);
    bool basic_vs_block = verifyResults(sum_basic, sum_block, n);

    // 输出结果
    cout << "基本逐行算法平均执行时间: " << time_basic << " 秒" << endl;
    cout << "循环展开版本平均执行时间: " << time_unroll << " 秒" << endl;
    cout << "分块优化版本平均执行时间: " << time_block << " 秒" << endl;

    cout << "加速比(循环展开/基本): " << time_basic / time_unroll << endl;
    cout << "加速比(分块优化/基本): " << time_basic / time_block << endl;

    // 检查结果
    cout << "结果验证: " 
         << (basic_vs_unroll ? "通过" : "失败") << " (基本与循环展开), "
         << (basic_vs_block ? "通过" : "失败") << " (基本与分块优化)" << endl;

    // 释放内存
    for (int i = 0; i < n; i++) {
        delete[] A[i];
        delete[] B[i];
    }
    delete[] A;
    delete[] B;
    delete[] sum_basic;
    delete[] sum_unroll;
    delete[] sum_block;

    return 0;
}
EOF
    echo "C++文件创建完成"
}

# 创建结果目录
mkdir -p $OUTPUT_DIR

# 编译不同优化级别的程序
compile_programs() {
    echo "使用ARM交叉编译器编译测试程序(静态链接)..."
    $ARM_CROSS_COMPILER -std=c++11 -O0 -static -o row_major_test_O0 row_major_benchmark.cpp
    $ARM_CROSS_COMPILER -std=c++11 -O1 -static -o row_major_test_O1 row_major_benchmark.cpp
    $ARM_CROSS_COMPILER -std=c++11 -O2 -static -o row_major_test_O2 row_major_benchmark.cpp
    $ARM_CROSS_COMPILER -std=c++11 -O3 -static -o row_major_test_O3 row_major_benchmark.cpp
    $ARM_CROSS_COMPILER -std=c++11 -O3 -mcpu=cortex-a72 -static -o row_major_test_O3_native row_major_benchmark.cpp
    echo "编译完成"
}

# 运行测试并保存结果
run_tests() {
    echo "运行QEMU模拟的ARM架构性能测试..."
    RESULT_FILE="$OUTPUT_DIR/performance_${TARGET_ARCH}_row_major.csv"

    # 创建CSV标题
    echo "架构,优化级别,矩阵大小,算法,执行时间(秒)" > $RESULT_FILE

    # 运行各种优化级别的测试
    for OPT in O0 O1 O2 O3 O3_native; do
        for SIZE in "${MATRIX_SIZES[@]}"; do
            echo "测试 $OPT 优化级别, 矩阵大小 $SIZE..."
            # 使用QEMU运行ARM二进制文件
            OUTPUT=$($QEMU_USER ./row_major_test_$OPT $SIZE $ITERATIONS)

            # 提取算法执行时间
            BASIC_TIME=$(echo "$OUTPUT" | grep "基本逐行算法平均执行时间" | awk '{print $2}')
            UNROLL_TIME=$(echo "$OUTPUT" | grep "循环展开版本平均执行时间" | awk '{print $2}')
            BLOCK_TIME=$(echo "$OUTPUT" | grep "分块优化版本平均执行时间" | awk '{print $2}')

            # 写入CSV
            echo "$TARGET_ARCH,$OPT,$SIZE,基本逐行,$BASIC_TIME" >> $RESULT_FILE
            echo "$TARGET_ARCH,$OPT,$SIZE,循环展开,$UNROLL_TIME" >> $RESULT_FILE
            echo "$TARGET_ARCH,$OPT,$SIZE,分块优化,$BLOCK_TIME" >> $RESULT_FILE

            # 保存完整输出
            echo "$OUTPUT" > "$OUTPUT_DIR/${TARGET_ARCH}_${OPT}_${SIZE}.txt"
        done
    done

    echo "性能测试完成，结果保存在 $RESULT_FILE"
}

# 创建结果可视化脚本
create_visualization_script() {
    echo "创建数据可视化Python脚本..."
    cat > visualize_row_major.py << 'EOF'
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def visualize_results(csv_file):
    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 确认数据列名
    print("CSV列名:", df.columns.tolist())

    # 获取不同的优化级别和算法
    opt_levels = df['优化级别'].unique()
    algorithms = df['算法'].unique()

    # 创建多个图表
    plt.figure(figsize=(15, 12))

    # 1. 矩阵大小与执行时间关系 (按优化级别 O3)
    plt.subplot(2, 2, 1)
    for alg in algorithms:
        data = df[(df['优化级别'] == 'O3') & (df['算法'] == alg)]
        plt.plot(data['矩阵大小'], data['执行时间(秒)'], marker='o', label=alg)

    plt.title('O3 Optimize the execution time of different algorithms')
    plt.xlabel('Matrix size')
    plt.ylabel('Execution time (seconds)')
    plt.grid(True)
    plt.legend()

    # 2. 优化级别对基本逐行算法的影响
    plt.subplot(2, 2, 2)
    for opt in opt_levels:
        data = df[(df['优化级别'] == opt) & (df['算法'] == '基本逐行')]
        plt.plot(data['矩阵大小'], data['执行时间(秒)'], marker='o', label=opt)

    plt.title('The influence of different optimization levels on the basic line-by-line algorithm')
    plt.xlabel('Matrix size')
    plt.ylabel('Execution time (seconds)')
    plt.grid(True)
    plt.legend()

    # 3. 加速比对比 (O3优化级别)
    plt.subplot(2, 2, 3)

    # 计算加速比
    opt = 'O3'
    sizes = []
    unroll_speedups = []
    block_speedups = []

    for size in sorted(df['矩阵大小'].unique()):
        basic_time = df[(df['优化级别'] == opt) & (df['算法'] == '基本逐行') & (df['矩阵大小'] == size)]['执行时间(秒)'].values
        unroll_time = df[(df['优化级别'] == opt) & (df['算法'] == '循环展开') & (df['矩阵大小'] == size)]['执行时间(秒)'].values
        block_time = df[(df['优化级别'] == opt) & (df['算法'] == '分块优化') & (df['矩阵大小'] == size)]['执行时间(秒)'].values

        if len(basic_time) > 0 and len(unroll_time) > 0 and len(block_time) > 0:
            sizes.append(size)
            unroll_speedups.append(basic_time[0] / unroll_time[0])
            block_speedups.append(basic_time[0] / block_time[0])

    plt.plot(sizes, unroll_speedups, marker='o', label='Loop expansion/Basic')
    plt.plot(sizes, block_speedups, marker='s', label='Block Optimization/Basic')

    plt.title('O3 Acceleration ratio at the optimal level')
    plt.xlabel('Matrix size')
    plt.ylabel('Acceleration ratio')
    plt.grid(True)
    plt.legend()

    # 4. 不同算法在不同优化级别的性能比较 (中等大小矩阵)
    plt.subplot(2, 2, 4)

    # 设置要比较的大小
    size_to_compare = 1000

    # 找到最接近的实际测试大小
    closest_size = df['矩阵大小'].unique()[np.abs(df['矩阵大小'].unique() - size_to_compare).argmin()]

    x_pos = np.arange(len(opt_levels))
    width = 0.25
    offsets = [-width, 0, width]

    for i, alg in enumerate(algorithms):
        times = []
        for opt in opt_levels:
            data = df[(df['优化级别'] == opt) & (df['算法'] == alg) & (df['矩阵大小'] == closest_size)]
            if not data.empty:
                times.append(data['执行时间(秒)'].values[0])
            else:
                times.append(0)

        plt.bar(x_pos + offsets[i], times, width, label=alg)

    plt.title(f'Matrix size {closest_size} Performance comparison of different optimization levels')
    plt.ylabel('Execution time (seconds)')
    plt.xticks(x_pos, opt_levels)
    plt.grid(True, axis='y')
    plt.legend()

    # 5. 添加第5个图: MCPU优化的影响
    plt.figure(figsize=(10, 6))

    algorithms = ['基本逐行', '分块优化']
    for alg in algorithms:
        data_o3 = df[(df['优化级别'] == 'O3') & (df['算法'] == alg)]
        data_native = df[(df['优化级别'] == 'O3_native') & (df['算法'] == alg)]

        if not data_o3.empty and not data_native.empty:
            plt.plot(data_o3['矩阵大小'], data_o3['执行时间(秒)'], marker='o', linestyle='-', label=f'{alg} O3')
            plt.plot(data_native['矩阵大小'], data_native['执行时间(秒)'], marker='s', linestyle='--', label=f'{alg} O3_native')

    plt.title('Performance impact of McPu-specific optimization (Cortex-A72)')
    plt.xlabel('Matrix size')
    plt.ylabel('Execution time (seconds)')
    plt.grid(True)
    plt.legend()

    # 调整布局并保存
    plt.tight_layout()
    output_file = os.path.splitext(csv_file)[0] + "_visualization.png"
    plt.savefig(output_file, dpi=300)

    # 保存第二个图表
    output_file2 = os.path.splitext(csv_file)[0] + "_mcpu_comparison.png"
    plt.figure(1).savefig(output_file2, dpi=300)

    print(f"结果图表已保存至 {output_file} 和 {output_file2}")

    try:
        plt.show()
    except:
        print("无法显示图表，可能是在没有GUI的环境中运行")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 visualize_row_major.py <csv_结果文件>")
        sys.exit(1)

    csv_file = sys.argv[1]
    if not os.path.exists(csv_file):
        print(f"错误: 找不到文件 {csv_file}")
        sys.exit(1)

    visualize_results(csv_file)
EOF
    echo "数据可视化脚本创建完成"
}

# 主流程
check_dependencies
create_cpp_file
compile_programs
run_tests
create_visualization_script

echo "所有测试完成！"
echo "模拟架构: $TARGET_ARCH"
echo "主机架构: $HOST_ARCH"
echo "测试结果目录: $OUTPUT_DIR"
echo "可使用以下命令可视化结果:"
echo "python3 visualize_row_major.py $OUTPUT_DIR/performance_${TARGET_ARCH}_row_major.csv"