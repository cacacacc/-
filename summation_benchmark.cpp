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
