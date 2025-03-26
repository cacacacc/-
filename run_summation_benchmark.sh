#!/bin/bash

# 设置编译选项
COMPILER="g++"
STD_VERSION="-std=c++11"
OPT_LEVELS=("O0" "O1" "O2" "O3")
SOURCE_FILE="summation_benchmark.cpp"
OUTPUT_DIR="summation_results"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 创建C++源文件
echo "创建 C++ 源文件: $SOURCE_FILE..."
cat > $SOURCE_FILE << 'EOF'
#include <iostream>
#include <vector>
#include <ctime>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>

using namespace std;
using namespace std::chrono;

// 链式求和算法
double chainSum(const vector<double>& a) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        sum += a[i];
    }
    return sum;
}

// 多链路求和算法
double multiChainSum(const vector<double>& a) {
    double sum1 = 0.0, sum2 = 0.0;
    size_t n = a.size();

    // 处理偶数部分
    for (size_t i = 0; i < n - (n % 2); i += 2) {
        sum1 += a[i];
        sum2 += a[i + 1];
    }

    // 处理最后一个元素（如果数组长度为奇数）
    if (n % 2 != 0) {
        sum1 += a[n - 1];
    }

    return sum1 + sum2;
}

// 递归方式求和算法 - 递归函数实现
double recursiveSum(vector<double>& a, size_t n) {
    if (n == 1) return a[0];

    for (size_t i = 0; i < n / 2; i++) {
        a[i] += a[n - i - 1];
    }

    return recursiveSum(a, (n + 1) / 2); // 上取整，处理奇数长度的情况
}

// 递归方式求和算法 - 循环实现
double iterativeRecursiveSum(vector<double> a) { // 传值，保持原数组不变
    size_t n = a.size();

    for (size_t m = n; m > 1; m = (m + 1) / 2) {
        for (size_t i = 0; i < m / 2; i++) {
            a[i] = a[i * 2] + a[i * 2 + 1];
        }

        // 如果元素个数为奇数，处理最后一个元素
        if (m % 2 != 0) {
            a[m / 2] = a[m - 1];
            m++;
        }
    }

    return a[0];
}

// 生成随机数据
vector<double> generateRandomData(size_t size) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-100.0, 100.0);

    vector<double> data(size);
    for (size_t i = 0; i < size; i++) {
        data[i] = dis(gen);
    }

    return data;
}

// 测量算法执行时间
template<typename Func, typename... Args>
pair<double, double> measureTime(Func func, Args&&... args) {
    auto start = high_resolution_clock::now();
    double result = func(forward<Args>(args)...);
    auto end = high_resolution_clock::now();

    duration<double, milli> time_ms = end - start;
    return {time_ms.count(), result};
}

int main(int argc, char* argv[]) {
    vector<size_t> sizes;

    // 检查命令行参数
    if (argc < 2) {
        // 默认测试大小
        sizes = {10, 100, 1000, 10000, 100000, 1000000, 10000000};
    } else {
        // 用户指定测试大小
        for (int i = 1; i < argc; i++) {
            sizes.push_back(stoul(argv[i]));
        }
    }

    // 输出表头
    cout << setw(12) << "数组大小" 
         << setw(15) << "链式(ms)" 
         << setw(15) << "多链路(ms)" 
         << setw(15) << "递归函数(ms)" 
         << setw(15) << "循环递归(ms)" 
         << setw(15) << "结果验证" << endl;

    // 对每个大小运行测试
    for (size_t size : sizes) {
        vector<double> data = generateRandomData(size);
        vector<double> data_copy = data; // 递归函数会修改原数组，需要拷贝

        // 测试链式求和
        auto [time_chain, result_chain] = measureTime(chainSum, data);

        // 测试多链路求和
        auto [time_multi, result_multi] = measureTime(multiChainSum, data);

        // 测试递归函数求和
        auto [time_rec, result_rec] = measureTime(recursiveSum, data_copy, data_copy.size());

        // 测试循环递归求和
        auto [time_iter, result_iter] = measureTime(iterativeRecursiveSum, data);

        // 验证结果一致性
        bool results_match = (fabs(result_chain - result_multi) < 1e-10) && 
                            (fabs(result_chain - result_rec) < 1e-10) && 
                            (fabs(result_chain - result_iter) < 1e-10);

        // 输出结果
        cout << setw(12) << size 
             << setw(15) << fixed << setprecision(3) << time_chain
             << setw(15) << fixed << setprecision(3) << time_multi
             << setw(15) << fixed << setprecision(3) << time_rec
             << setw(15) << fixed << setprecision(3) << time_iter
             << setw(15) << (results_match ? "通过" : "失败") << endl;
    }

    return 0;
}
EOF

# 定义测试数据大小
SIZES=(10 100 1000 10000 100000 1000000 10000000)

# 编译不同优化级别的程序
for OPT in "${OPT_LEVELS[@]}"; do
    echo "使用 -$OPT 优化级别编译..."
    $COMPILER $STD_VERSION -$OPT -o summation_benchmark_$OPT $SOURCE_FILE

    # 检查编译是否成功
    if [ $? -ne 0 ]; then
        echo "编译失败，退出"
        exit 1
    fi

    # 运行测试
    echo "运行 -$OPT 优化级别的测试..."
    RESULT_FILE="$OUTPUT_DIR/summation_results_$OPT.txt"

    # 将测试结果保存到文件
    ./summation_benchmark_$OPT "${SIZES[@]}" | tee $RESULT_FILE
done

# 创建可视化脚本
echo "创建数据可视化Python脚本..."
cat > visualize_summation.py << 'EOF'
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
import os


def parse_results(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 跳过标题行
    data = []
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) >= 5:
            size = int(parts[0])
            chain_time = float(parts[1])
            multi_time = float(parts[2])
            recursive_time = float(parts[3])
            iterative_time = float(parts[4])
            data.append((size, chain_time, multi_time, recursive_time, iterative_time))

    return data

def extract_opt_level(filename):
    match = re.search(r'_O(\d)\.txt$', filename)
    if match:
        return f"O{match.group(1)}"
    return "Unknown"

def visualize_results():
    # 获取所有结果文件
    result_files = glob.glob("summation_results/*.txt")

    if not result_files:
        print("没有找到结果文件")
        return

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('求和算法性能对比', fontsize=16)

    # 颜色和标记
    colors = ['b', 'g', 'r', 'c']
    markers = ['o', 's', '^', 'd']

    # 为每个优化级别创建子图
    for idx, file_path in enumerate(sorted(result_files)):
        opt_level = extract_opt_level(file_path)
        data = parse_results(file_path)

        # 提取数据
        sizes = [d[0] for d in data]
        chain_times = [d[1] for d in data]
        multi_times = [d[2] for d in data]
        recursive_times = [d[3] for d in data]
        iterative_times = [d[4] for d in data]

        # 计算加速比
        speedup_multi = [chain / multi if multi > 0 else 1 for chain, multi in zip(chain_times, multi_times)]
        speedup_recursive = [chain / rec if rec > 0 else 1 for chain, rec in zip(chain_times, recursive_times)]
        speedup_iterative = [chain / it if it > 0 else 1 for chain, it in zip(chain_times, iterative_times)]

        # 绘制执行时间图 (对数尺度)
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        ax.set_title(f'Optimization level: {opt_level}')
        ax.set_xlabel('Array size')
        ax.set_ylabel('Execution time (ms, logarithmic scale)')
        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.plot(sizes, chain_times, color=colors[0], marker=markers[0], label='链式')
        ax.plot(sizes, multi_times, color=colors[1], marker=markers[1], label='多链路')
        ax.plot(sizes, recursive_times, color=colors[2], marker=markers[2], label='递归函数')
        ax.plot(sizes, iterative_times, color=colors[3], marker=markers[3], label='循环递归')

        ax.grid(True, which="both", ls="--")
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('summation_results/execution_times.png', dpi=300)

    # 创建加速比图表
    plt.figure(figsize=(12, 8))
    plt.title('Acceleration ratio of different algorithms to chain summation (O3 optimization)', fontsize=14)
    plt.xlabel('Array size')
    plt.ylabel('Acceleration ratio')
    plt.xscale('log')

    # 使用O3优化级别的数据
    o3_file = next((f for f in result_files if 'O3.txt' in f), None)
    if o3_file:
        data = parse_results(o3_file)
        sizes = [d[0] for d in data]
        chain_times = [d[1] for d in data]
        multi_times = [d[2] for d in data]
        recursive_times = [d[3] for d in data]
        iterative_times = [d[4] for d in data]

        speedup_multi = [chain / multi if multi > 0 else 1 for chain, multi in zip(chain_times, multi_times)]
        speedup_recursive = [chain / rec if rec > 0 else 1 for chain, rec in zip(chain_times, recursive_times)]
        speedup_iterative = [chain / it if it > 0 else 1 for chain, it in zip(chain_times, iterative_times)]

        plt.plot(sizes, speedup_multi, color='g', marker='s', label='multilink')
        plt.plot(sizes, speedup_recursive, color='r', marker='^', label='Recursive function')
        plt.plot(sizes, speedup_iterative, color='c', marker='d', label='Cyclic recursion')
        plt.axhline(y=1, color='b', linestyle='--', label='Chain (reference)')

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('summation_results/speedup_ratios.png', dpi=300)

    print("可视化图表已保存至 summation_results 目录")

    try:
        plt.show()
    except:
        print("无法显示图表，可能是在没有GUI的环境中运行")

if __name__ == "__main__":
    visualize_results()
EOF

echo "运行Python可视化脚本..."
python3 visualize_summation.py

echo "============================================="
echo "所有测试完成！"
echo "测试结果保存在: $OUTPUT_DIR/"
echo "可视化图表保存在: $OUTPUT_DIR/"
echo "============================================="