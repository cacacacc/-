#!/bin/bash

# 设置测试参数
MATRIX_SIZES=(100 200 300 400 500 600 700 800 900 1000 1100 1200 1300
1400 1500 1600 1700 1800 1900 2000 2100 2200 2300 2400 2500 2600 2700 
2800 2900 3000 3100 3200 3300 3400 3500 3600 3700 3800 3900 4000)
ITERATIONS=10
TARGET_ARCH="arm64"
HOST_ARCH=$(uname -m)
OUTPUT_DIR="results_${TARGET_ARCH}_emulated"

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

# 创建C++测试程序文件
create_cpp_file() {
    echo "创建矩阵测试C++文件..."
    cat > matrix_benchmark.cpp << 'EOF'
#include <iostream>
#include <ctime>
#include <vector>
#include <iomanip>
#include <cstring>
#include <numeric>
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

// 算法1: 逐列访问 (Cache不友好)
void columnMajorSum(double** A, double** B, double* sum, int n) {
    for (int i = 0; i < n; i++) {
        sum[i] = 0.0;
        for (int j = 0; j < n; j++) {
            sum[i] += B[j][i] * A[j][i];  // 逐列访问，导致Cache Miss率高
        }
    }
}

// 算法1: 循环展开版本
void columnMajorSumUnrolled(double** A, double** B, double* sum, int n) {
    for (int i = 0; i < n; i++) {
        sum[i] = 0.0;
    }

    for (int i = 0; i < n; i++) {
        int j = 0;
        for (; j < n-3; j += 4) {
            sum[i] += B[j][i] * A[j][i];
            sum[i] += B[j+1][i] * A[j+1][i];
            sum[i] += B[j+2][i] * A[j+2][i];
            sum[i] += B[j+3][i] * A[j+3][i];
        }
        // 处理剩余元素
        for (; j < n; j++) {
            sum[i] += B[j][i] * A[j][i];
        }
    }
}

// 算法2: 逐行访问 (Cache友好)
void rowMajorSum(double** A, double** B, double* sum, int n) {
    memset(sum, 0, n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            sum[j] += B[i][j] * A[i][j];  // 逐行访问，适合缓存局部性
        }
    }
}

// 算法2: 循环展开版本
void rowMajorSumUnrolled(double** A, double** B, double* sum, int n) {
    memset(sum, 0, n * sizeof(double));

    for (int i = 0; i < n; i++) {
        int j = 0;
        for (; j < n-3; j += 4) {
            sum[j] += B[i][j] * A[i][j];
            sum[j+1] += B[i][j+1] * A[i][j+1];
            sum[j+2] += B[i][j+2] * A[i][j+2];
            sum[j+3] += B[i][j+3] * A[i][j+3];
        }
        // 处理剩余元素
        for (; j < n; j++) {
            sum[j] += B[i][j] * A[i][j];
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

    double* sum1 = new double[n];
    double* sum1_unroll = new double[n];
    double* sum2 = new double[n];
    double* sum2_unroll = new double[n];

    // 初始化数据
    initializeMatrix(A, B, n);

    // 测量性能
    double time_alg1 = measureTime(columnMajorSum, A, B, sum1, n, repeat);
    double time_alg1_unroll = measureTime(columnMajorSumUnrolled, A, B, sum1_unroll, n, repeat);
    double time_alg2 = measureTime(rowMajorSum, A, B, sum2, n, repeat);
    double time_alg2_unroll = measureTime(rowMajorSumUnrolled, A, B, sum2_unroll, n, repeat);

    // 验证结果一致性
    bool alg1_vs_alg2 = verifyResults(sum1, sum2, n);
    bool alg1_vs_alg1_unroll = verifyResults(sum1, sum1_unroll, n);
    bool alg2_vs_alg2_unroll = verifyResults(sum2, sum2_unroll, n);

    // 输出结果
    cout << "算法1平均执行时间: " << time_alg1 << " 秒" << endl;
    cout << "算法1循环展开版本平均执行时间: " << time_alg1_unroll << " 秒" << endl;
    cout << "算法2平均执行时间: " << time_alg2 << " 秒" << endl;
    cout << "算法2循环展开版本平均执行时间: " << time_alg2_unroll << " 秒" << endl;

    cout << "加速比(算法2/算法1): " << time_alg1 / time_alg2 << endl;
    cout << "加速比(算法1展开/算法1): " << time_alg1 / time_alg1_unroll << endl;
    cout << "加速比(算法2展开/算法2): " << time_alg2 / time_alg2_unroll << endl;

    // 检查结果
    cout << "结果验证: " 
         << (alg1_vs_alg2 ? "通过" : "失败") << " (算法1与算法2), "
         << (alg1_vs_alg1_unroll ? "通过" : "失败") << " (算法1与其展开版本), "
         << (alg2_vs_alg2_unroll ? "通过" : "失败") << " (算法2与其展开版本)" << endl;

    // 释放内存
    for (int i = 0; i < n; i++) {
        delete[] A[i];
        delete[] B[i];
    }
    delete[] A;
    delete[] B;
    delete[] sum1;
    delete[] sum1_unroll;
    delete[] sum2;
    delete[] sum2_unroll;

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
    $ARM_CROSS_COMPILER -std=c++11 -O0 -static -o matrix_test_O0 matrix_benchmark.cpp
    $ARM_CROSS_COMPILER -std=c++11 -O1 -static -o matrix_test_O1 matrix_benchmark.cpp
    $ARM_CROSS_COMPILER -std=c++11 -O2 -static -o matrix_test_O2 matrix_benchmark.cpp
    $ARM_CROSS_COMPILER -std=c++11 -O3 -static -o matrix_test_O3 matrix_benchmark.cpp
    $ARM_CROSS_COMPILER -std=c++11 -O3 -mcpu=cortex-a72 -static -o matrix_test_O3_native matrix_benchmark.cpp
    echo "编译完成"
}

# 运行测试并保存结果
run_tests() {
    echo "运行QEMU模拟的ARM架构性能测试..."
    RESULT_FILE="$OUTPUT_DIR/performance_${TARGET_ARCH}_emulated.csv"

    # 创建CSV标题
    echo "架构,优化级别,矩阵大小,算法,执行时间(秒)" > $RESULT_FILE

    # 运行各种优化级别的测试
    for OPT in O0 O1 O2 O3 O3_native; do
        for SIZE in "${MATRIX_SIZES[@]}"; do
            echo "测试 $OPT 优化级别, 矩阵大小 $SIZE..."
            # 使用QEMU运行ARM二进制文件
            OUTPUT=$($QEMU_USER ./matrix_test_$OPT $SIZE $ITERATIONS)

            # 提取算法执行时间
            ALG1_TIME=$(echo "$OUTPUT" | grep "算法1平均执行时间" | awk '{print $2}')
            ALG1_UNROLL_TIME=$(echo "$OUTPUT" | grep "算法1循环展开版本平均执行时间" | awk '{print $2}')
            ALG2_TIME=$(echo "$OUTPUT" | grep "算法2平均执行时间" | awk '{print $2}')
            ALG2_UNROLL_TIME=$(echo "$OUTPUT" | grep "算法2循环展开版本平均执行时间" | awk '{print $2}')

            # 写入CSV
            echo "$TARGET_ARCH,$OPT,$SIZE,算法1,$ALG1_TIME" >> $RESULT_FILE
            echo "$TARGET_ARCH,$OPT,$SIZE,算法1_展开,$ALG1_UNROLL_TIME" >> $RESULT_FILE
            echo "$TARGET_ARCH,$OPT,$SIZE,算法2,$ALG2_TIME" >> $RESULT_FILE
            echo "$TARGET_ARCH,$OPT,$SIZE,算法2_展开,$ALG2_UNROLL_TIME" >> $RESULT_FILE

            # 保存完整输出
            echo "$OUTPUT" > "$OUTPUT_DIR/${TARGET_ARCH}_${OPT}_${SIZE}.txt"
        done
    done

    echo "性能测试完成，结果保存在 $RESULT_FILE"
}

# 创建结果可视化脚本
create_visualization_script() {
    echo "创建数据可视化Python脚本..."
    cat > visualize_results.py << 'EOF'
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
    plt.figure(figsize=(15, 10))

    # 1. 矩阵大小与执行时间关系 (按优化级别)
    plt.subplot(2, 2, 1)
    for opt in opt_levels:
        for alg in ['算法2']: # 只显示行优先算法
            data = df[(df['优化级别'] == opt) & (df['算法'] == alg)]
            plt.plot(data['矩阵大小'], data['执行时间(秒)'], marker='o', label=f'{opt}-{alg}')

    plt.title('Effect of matrix size on execution time (by optimization level)')
    plt.xlabel('Matrix size')
    plt.ylabel('Execution time (seconds)')
    plt.grid(True)
    plt.legend()

    # 2. 矩阵大小与执行时间关系 (按算法)
    plt.subplot(2, 2, 2)
    for alg in algorithms:
        data = df[(df['优化级别'] == 'O3') & (df['算法'] == alg)]
        plt.plot(data['矩阵大小'], data['执行时间(秒)'], marker='o', label=alg)

    plt.title('Effect of matrix size on execution time (O3 optimization, by algorithm)')
    plt.xlabel('Matrix size')
    plt.ylabel('Execution time (seconds)')
    plt.grid(True)
    plt.legend()

    # 3. 算法加速比 (算法2/算法1)
    plt.subplot(2, 2, 3)
    for opt in opt_levels:
        # 计算加速比
        alg1_data = df[(df['优化级别'] == opt) & (df['算法'] == '算法1')]
        alg2_data = df[(df['优化级别'] == opt) & (df['算法'] == '算法2')]

        if not alg1_data.empty and not alg2_data.empty:
            # 确保数据顺序相同
            alg1_data = alg1_data.sort_values('矩阵大小')
            alg2_data = alg2_data.sort_values('矩阵大小')

            sizes = alg1_data['矩阵大小'].values
            speedups = alg1_data['执行时间(秒)'].values / alg2_data['执行时间(秒)'].values

            plt.plot(sizes, speedups, marker='o', label=opt)

    plt.title('The speed ratio of row access to column access')
    plt.xlabel('Matrix size')
    plt.ylabel('Acceleration ratio (Algorithm 2/ Algorithm 1)')
    plt.grid(True)
    plt.legend()

    # 4. 优化级别比较
    plt.subplot(2, 2, 4)

    # 设置要比较的大小
    size_to_compare = 1000

    # 找到最接近的实际测试大小
    closest_size = df['矩阵大小'].unique()[np.abs(df['矩阵大小'].unique() - size_to_compare).argmin()]

    x_pos = np.arange(len(opt_levels))
    width = 0.2
    offsets = [-0.3, -0.1, 0.1, 0.3]

    for i, alg in enumerate(algorithms):
        times = []
        for opt in opt_levels:
            data = df[(df['优化级别'] == opt) & (df['算法'] == alg) & (df['矩阵大小'] == closest_size)]
            if not data.empty:
                times.append(data['执行时间(秒)'].values[0])
            else:
                times.append(0)

        plt.bar(x_pos + offsets[i], times, width, label=alg)

    plt.title(f'Comparison of execution times for different optimization levels (matrix size {closest_size})')
    plt.ylabel('Execution time (seconds)')
    plt.xticks(x_pos, opt_levels)
    plt.grid(True, axis='y')
    plt.legend()

    # 调整布局并保存
    plt.tight_layout()
    output_file = os.path.splitext(csv_file)[0] + "_visualization.png"
    plt.savefig(output_file, dpi=300)
    print(f"结果图表已保存至 {output_file}")

    try:
        plt.show()
    except:
        print("无法显示图表，可能是在没有GUI的环境中运行")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 visualize_results.py <csv_结果文件>")
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
echo "python3 visualize_results.py $OUTPUT_DIR/performance_${TARGET_ARCH}_emulated.csv"