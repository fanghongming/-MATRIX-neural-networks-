# -MATRIX-neural-networks-
矩阵和神经网络库（MATRIX  neural networks）
我们使用的是Eigen库，想要这个库，我们可以去https://eigen.tuxfamily.org/index.php?title=Main_Page
这个地方有文档，也可以在github上找到,https://github.com/PX4/eigen。

我演示用vscode展示如何

![image](https://github.com/user-attachments/assets/262c246a-0764-436c-8de1-5a063205d89f)

![image](https://github.com/user-attachments/assets/3ddcb833-ce35-4177-bda6-b700de8234de)

然后直接导入库就可以了，注意命名空间。
``` cpp
#include <Eigen/Dense>
```

教学资料在https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html

# “空不是无，空是一种存在，你得用空这种存在填满自己。”

## 一个基本的代数变形

我们先学习一个递归的等式。

这个等式告诉我们的是，如何递归求解期望，类似于霍纳法则。

已知问题：不用任何数组，解决一组数据的平均值的问题。

我们知道

$$E[x]=\lim_{n\to \infty}\frac{\sum_{i=1}^nx_i}{n}$$

设$w_{k}$是第k次后求出来的平均值，设$w_{k+1}$是第k+1次求出来的平均值。

数学之间的问题就是发现之间的关系。

我们发现

$$w_{k+1}=\frac{1}{k+1}\sum_{i=1}^{k+1}x_i,k=1,2,3$$

并且

$$w_{k}=\frac{1}{k}\sum_{i=1}^kx_i,k=1,2,3$$

通过观察，我们发现

$$\begin{align*}
&w_{k+1}=\frac{1}{k+1}\sum_{i=1}^{k+1}x_i=\frac{1}{k+1}(\sum_{i=1}^{k}x_i+x_k)
\\&=\frac{1}{k+1}(kw_k+x_k)
\\&=w_k-\frac{1}{k+1}(w_k-x_k)
\end{align*}$$

这个形式是一个十分优雅的结构，至少有一个特点，就是可以忘记前面的所有经验。

## 随机梯度下降算法

我们考虑到这样一个最优问题,f是一个函数，设X是随机变量(一个将事件映射到实数的函数),它代表的是一种经验,$w$是模型的参数,可能是一个向量，也可能是标量，你能调整的是$w$

$$\min_{w}J(w)=E[f(w,X)]$$

根据梯度的性质和期望的性质，我们有

$$\begin{align*}
&\nabla _wE[f(w,X)]=\nabla_w\sum_{i}p_if(w,x_i)
\\&=\sum_i p_i\nabla_wf(w,x_i)(因为函数之和的梯度等于函数梯度之和)
\\&=E[\nabla_{w}f(w,X)]
\end{align*}$$

![](https://files.mdnice.com/user/72186/f56090fe-94d9-4a61-bc6b-2dfa674472a7.png)

这个$E[\nabla _{w}f(w,X)]$最直观的想法就是多跑几次蒙特卡洛，求解出来。
$$E[\nabla _w f(w,X)]=\lim_{n\to \infty}\frac{1}{n}\sum_{i=1}^n \nabla_wf(w,x_i) $$

我们设第k次迭代更新后的参数值为$w_k$,那么我们通过前面的这个公式推导出来我们有

$$w_{k+1}=w_k-\alpha_kJ(w_k)=w_k-\alpha_kE[\nabla _wf(w_k,X)]
$$
$$=w_k-\frac{\alpha_k}{n}\sum_{i=1}^n\nabla_w f(w_k,x_i)$$
## 举个例子


![](https://files.mdnice.com/user/72186/c0fd2c87-846b-4801-88db-2ba0bd3482ad.png)


我们接下来，用一个高中的知识，就是所谓的最小二乘法能解决的问题，我们用随机梯度下降算法解决一下。

想到这，笔者还想到自己的一个故事，因为我们当时高三还用老教材，而笔者用新教材，被这个最小二乘法的证明吸引，当时缺少对于和式$\sum$这个玩意的背景，攻克下来后，当时就觉得十分有趣，没想到这个经常作为机器学习的案例。
### 定义损失函数

$$\begin{align*}
&对于单个样本(x,y)，线性回归模型的预测值为\hat{y}=w\times x+b
\\&均方误差损失函数L(w,b) 定义为预测值和真实值之差的平方的一半：
\\&L(w,b)=\frac{1}{2}(\hat{y}-y)^2=\frac{1}{2}(w\times x+b-y)^2
\\&这里乘以\frac{1}{2}是为了在求导时消除常数系数，方便后续计算，不影响最终的优化结果。


\end{align*}$$

### 关于权重w的梯度
$$
\begin{align*}
&计算损失函数关于权重w的梯度\frac{\partial L(w,b)}{\partial w}
\\&设u=w\times x+b-y,那么L(w,b)=\frac{1}{2}u^2
\\&先对L(u)关于u求导:
\\&\frac{\partial L(u)}{\partial u}=u=w\times x+b-y
\\&在对u关于w求导:
\\&\frac{\partial u}{\partial w}=x
\\&根据复合函数求导公式\frac{\partial L(w,b)}{\partial  w}
\\&=\frac{\partial L(u)}{\partial  u}\times \frac{\partial u}{\partial w},可得
\\&\frac{\partial L(w,b)}{\partial w}=(w\times x+b-y)\times x
\\&=(\hat{y}-y)\times x
\end{align*}
$$

### 关于偏置 b的梯度

$$\begin{align*}
&计算损失函数关于b的梯度 \frac{\partial L(w,b)}{\partial b}
\\&同样设u=w\times x+b-y,那么L(w,b)=\frac{1}{2}u^2
\\&首先L(u)关于u求导:
\\&\frac{\partial L(u)}{\partial u}=u=w\times x+b-y
\\&再对u关于b 求导:
\\&\frac{\partial u}{\partial b}=1
\\&根据复合函数求导公式\frac{\partial L(w,b)}{\partial b}
\\&=\frac{\partial L(u)}{\partial u}\frac{\partial u}{\partial b},可得
\\&\frac{\partial L(w,b)}{\partial b}=w\times x+b-y
\\&=\hat{y}-y 
\end{align*}$$


```
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
using namespace std;
// 线性回归模型：y = w * x + b
struct Model {
    double w;  // 权重
    double b;  // 偏置
};

// 计算单个样本的预测值
double predict(const Model& model, double x) {
    return model.w * x + model.b;
}

// 计算单个样本的损失梯度（MSE损失）
void compute_gradient(const Model& model, double x, double y, 
                      double& dw, double& db) {
    double error = predict(model, x) - y;  // 预测值与真实值的差
    dw = error * x;  // 权重梯度 = 误差 * 输入x
    db = error;      // 偏置梯度 = 误差
}

// 随机梯度下降更新参数
void sgd_update(Model& model, double learning_rate, 
                double dw, double db) {
    model.w -= learning_rate * dw;
    model.b -= learning_rate * db;
}

// 生成模拟数据（线性关系 + 噪声）
std::vector<std::pair<double, double>> generate_data(int num_samples) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0, 0.5);
    
    std::vector<std::pair<double, double>> data;
    for (int i = 0; i < num_samples; ++i) {
        double x = static_cast<double>(i) / num_samples;
        double y = -2.0 * x + 2.0 + noise(gen);  // 真实模型: y = 3x + 2
        data.emplace_back(x, y);
    }
    return data;
}

int main() {
    // 超参数设置
    const double learning_rate = 0.1;
    const int epochs = 1000;
    const int num_samples = 100;

    // 初始化模型参数
    Model model{0.0, 0.0};  // 初始值 w=0, b=0

    // 生成模拟数据
    auto data = generate_data(num_samples);

    // 随机数生成器用于采样
    std::random_device rd;
    std::mt19937 gen(rd());

    // SGD 训练循环
    for (int epoch = 0; epoch < epochs*10; ++epoch) {
        // 随机打乱数据顺序
        for(int i=0;i<data.size();i++)
        {
            swap(data[i],data[(rand()%(data.size()-1-i+1)+i)]);
        }
        // 遍历所有样本（此处为纯随机梯度下降，逐样本更新）
        for (const auto& [x, y] : data) {
            double dw, db;
            compute_gradient(model, x, y, dw, db);
            sgd_update(model, learning_rate, dw, db);
        }

        // 每隔一定轮数输出损失
        if (epoch % 100 == 0) {
            double total_loss = 0;
            for (const auto& [x, y] : data) {
                double error = predict(model, x) - y;
                total_loss += error * error;
            }
            std::cout << "Epoch " << epoch 
                      << ", Loss: " << total_loss / num_samples 
                      << ", w=" << model.w << ", b=" << model.b << std::endl;
        }
    }

    // 输出最终参数
    std::cout << "\nFinal model: y = " << model.w << "x + " << model.b << std::endl;
    return 0;
}
```
### 可以用公式求解但计算量过大的情况
- 在许多机器学习和深度学习问题中，目标函数（如损失函数）关于模型参数的梯度在理论上是可以通过求导公式计算出来的。例如，对于线性回归模型的均方误差损失函数，其梯度有明确的解析表达式。但是，当数据量非常大时，计算整个数据集上的梯度会涉及到对大量数据的求和等操作，计算量巨大，耗时很长。
- 以逻辑回归模型在大规模数据集上的训练为例，如果使用传统的梯度下降算法，每次更新参数都需要遍历整个数据集来计算梯度，当数据集达到数百万甚至数亿条记录时，计算一次梯度可能需要花费很长时间，导致模型训练效率极低。此时，随机梯度下降就可以发挥作用，它每次只使用一个或一小批样本（即mini-batch）来近似估计梯度，大大减少了计算量，加快了训练速度。

### 难以用公式求解的情况
- 有些复杂的模型或问题，目标函数可能非常复杂，难以直接通过求导等方法得到梯度的解析表达式。例如，在一些涉及到复杂的神经网络结构、非凸优化问题或者存在隐变量的模型中，精确计算梯度可能非常困难甚至无法实现。
- 比如深度神经网络中的一些复杂的激活函数组合、递归神经网络（RNN）或长短期记忆网络（LSTM）在处理序列数据时，由于模型结构的复杂性和数据的动态性，很难直接写出梯度的精确公式。这时，随机梯度下降及其变种（如Adagrad、Adadelta、Adam等）可以通过数值近似等方法来估计梯度，从而实现模型的训练。

所以，随机梯度下降的应用场景主要是在梯度计算困难或计算成本过高的情况下，但并不意味着梯度一定不能用公式求解。它是一种为了更高效地进行模型训练而采用的优化算法，通过对梯度的近似估计来快速找到目标函数的最优解或近似最优解。

