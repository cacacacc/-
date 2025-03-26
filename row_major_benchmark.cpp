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
