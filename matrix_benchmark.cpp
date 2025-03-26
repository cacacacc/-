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
