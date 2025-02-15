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

# 福兮祸之所倚，祸兮福之所伏

反向传播算法是应用在前馈神经网络，它的目的是训练一个人工神经网络，这个基于仿生学的一种算法，但是更是基于数学的。

BP算法最重要的关键，也就是比较难以理解的地方，就是公式的多样性和参数的多样性。

但是实际上只是应用到基础的知识。
![](https://files.mdnice.com/user/72186/9a7baa85-7b4a-4419-adf7-1fc215effafe.png)

感觉这门学科正在变成一个经验科学，类似生物，化学，当然，其可解释性还需要去研究。

神经网络是一个非常大的函数，这个函数返回一些值，当然不一定是正确的，但是你可以调。

 我们先规定一些变量。
![](https://files.mdnice.com/user/72186/dc0f4e34-e857-4d93-afe7-a4ebfaa4e2e8.png)


希望大家脑子里神经网络能看懂这个东西 .


设总层数(包括输入输出和隐含)为d,输入层的大小为$sz_1$,输出层的大小为$sz_d$
我们想要求解误差，我们一般不直接用一次方误差，因为比较难算（笔者的高中数学老师告诉，大概意思是如果求解一次方，那么我们就必须要求解绝对值，难算，还有更深的原因，就是平方是光滑的，所以高中讲最小二乘法。）

对一每次训练，我们定义最终神经网络的输出为$\vec{y'}=(y_1',y_2',...,y_{sz_d}')$

这就是其中

$$\begin{align*}
&设\sigma(x)=\frac{1}{1+e^{-x}}
\\&我们希望求解\arg _{w}E_k=\frac{1}{2}  \sum_{i=1}^{sz_{d}}((\sigma(\sum_{j=1}^{sz_{d-1}}w^{d-1}_{ji}\times a^{d-1}+b_{i}^d))-y_j)^2

\end{align*}$$

总体公式的展开就是如此，我们不妨压缩公式，分层次。

我们设

$$z_i^d=\beta_{i}^d+b_{i}^d$$

$$\beta_i^d=\sum_{j=1}^{sz_{d-1}}w_{ji}^{d-1}a_{j}^{d-1}$$
$$f_j^d=\sigma _{}(z_i^d)-y_i$$
我们的公式可以压缩成

$$arg_w\sum_{i=1}^{sz_d}(f_j^d)^2$$

根据梯度下降算法，我们想要找的就是$w和b$的梯度，于是我们固定某些值，这些值就是$\vec{y}$，就像求解高中题一样固定某个函数中的参数，因为你有理由认为这个y就是正确的经验。 

如果按照《高等数学》的教法，那么就是求出某个数的梯度，梯度的解就是根据偏导数。
求
全面考虑各个维度上的共吸纳，各个参数给出的贡献，就是偏导数想要告诉我们的。

如果只是二维的空间的导数，不全面。

容易看到如下公式
$$\frac{\partial E}{\partial w_{ik}^d}=\frac{\partial E}{\partial f_j^d}\frac{\partial f_j^d}{\partial z_i^d }\frac{ \partial z_i^d }{\partial w_{ik}^d}$$

当然也不容易看到。当然，一般高等数学课程都会教一些内容，比如宋浩老师啊，或者你的数学老师。

![](https://files.mdnice.com/user/72186/d61d32ea-729a-4966-a432-6f5d50cf4b84.png)

其中
$$\frac{\partial E}{\partial f_j^d}=f_j^d$$
我们将对$\frac{1}{1+e^{-x}}$求导，我们发现

$$(\frac{1}{1+e^{-x}})'=-\frac{1}{(1+e^{-x})^2}(e^{-x})$$




$$=-\frac{e^{-x}}{(1+e^{-x})^2}=\frac{1+e^{-x}-1}{(1+e^{-x})^2}$$
$$=\sigma(x)(1-\sigma(x))$$
$$\frac{\partial f_j^d}{\partial z_i^d} =\sigma (z_i^d)(1-\sigma (z_i^d))$$

$$\frac{z_i^d}{\partial w_{ik}^d}=a_i^{d-1}$$

然后，我们讲所有东西带入，就求出了关于这个函数在$w_i$上的梯度。

$$\frac{\partial E}{\partial b_d^i}=\frac{\partial E}{\partial f_j^d}\frac{\partial f_j^d}{\partial z_i^d }\frac{ \partial z_i^d }{\partial b_d^i}$$

很容易看到$\frac{ \partial z_i^d }{\partial b_d^i}=1$

那么还有一个求导就是

$$\frac{\partial E}{\partial a_{i}^d}$$

这个稍微有点复杂，因为这是唯一需要和式的，也就是我们可以看这元素的多次出现。
![](https://files.mdnice.com/user/72186/4b306844-ec9e-4cce-9fcb-3767208441a6.png)

$$\frac{\partial E}{\partial a_i^{d-1}}=\sum_{j=1}^{sz_d}\frac{\partial E}{\partial f_{j}^d}\frac{\partial f_j^d}{\partial z_j^d}\frac{\partial z_j^d}{\partial a_{i}^{d-1}}$$
其中
$$\frac{\partial z_j^d}{\partial a_{i}^{d-1}}=w_{ij}^d$$

有了这个值，我们就可以向前递归出前面的值！

 $$\frac{\partial E}{\partial w_{ij}^{d-2}}=\frac{\partial E}{\partial a_{j}^{d-1}}\frac{\partial a_j^{d-1}}{\partial w_{ij}^{d-2}}$$
其中$\frac{\partial E}{\partial a_{j}^{d-1}}$已经求解出来了，$\frac{\partial a_j^{d-1}}{\partial w_{ij}^{d-2}}=a_i^{d-2}$
 $$\frac{\partial E}{\partial b_{i}^{d-2}}=\frac{\partial E}{\partial a_{j}^{d-1}}\frac{\partial a_j^{d-1}}{\partial b_i^{d-2}}$$
 
 对于$\frac{\partial E}{\partial a_{i}^{d-2}}$,因为$a_{i}^{d-2}$能影响很多变量，所以
 
 $$\frac{\partial E}{\partial a_{i}^{d-2}}=\sum_{j=1}^{sz_{d-1}}\frac{\partial E}{\partial a_j^{d-1}}\frac{\partial a_j^{d-1}}{\partial z_j^{d-2}}\frac{\partial z_j^{d-2}}{\partial a_i^{d-2}}$$
 
其中$\frac{\partial  a_j^{d-1}}{\partial z_j^{d-1}}=\sigma (z_j^{d-1})(1-\sigma (z_j^{d-1})$,$\frac{\partial z_j^{d-1}}{\partial a_i^{d-2}}=w_{ij}^{d-2}$
 
实际上就是动态规划的思想，递归的思想。

顺便放上代码，我们使用的库就是eigen。
``` cpp
#include <iostream>
#include <Eigen/Dense>
#include <cmath>

// 定义 sigmoid 激活函数
Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& x) {
    return 1.0 / (1.0 + (-x).array().exp());
}

// 定义 sigmoid 激活函数的导数
Eigen::MatrixXd sigmoid_derivative(const Eigen::MatrixXd& x) {
    return x.array() * (1 - x.array());
}

// 神经网络类
class NeuralNetwork {
private:
    int input_size;
    int hidden_size;
    int output_size;
    Eigen::MatrixXd weights_ih;  // 输入层到隐藏层的权重
    Eigen::MatrixXd weights_ho;  // 隐藏层到输出层的权重
    Eigen::VectorXd bias_h;      // 隐藏层的偏置
    Eigen::VectorXd bias_o;      // 输出层的偏置

public:
    // 构造函数，初始化网络参数
    NeuralNetwork(int input_size, int hidden_size, int output_size)
        : input_size(input_size), hidden_size(hidden_size), output_size(output_size) {
        // 随机初始化权重和偏置
        weights_ih = Eigen::MatrixXd::Random(hidden_size, input_size);
        weights_ho = Eigen::MatrixXd::Random(output_size, hidden_size);
        bias_h = Eigen::VectorXd::Random(hidden_size);
        bias_o = Eigen::VectorXd::Random(output_size);
    }

    // 前向传播
    Eigen::MatrixXd feedforward(const Eigen::MatrixXd& input) {
        // 计算隐藏层的输入
        Eigen::MatrixXd hidden_input = weights_ih * input + bias_h.replicate(1, input.cols());
        // 应用 sigmoid 激活函数
        Eigen::MatrixXd hidden_output = sigmoid(hidden_input);

        // 计算输出层的输入
        Eigen::MatrixXd output_input = weights_ho * hidden_output + bias_o.replicate(1, hidden_output.cols());
        // 应用 sigmoid 激活函数
        Eigen::MatrixXd output = sigmoid(output_input);

        return output;
    }

    // 训练函数
    void train(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target, double learning_rate) {
        // 前向传播
        Eigen::MatrixXd hidden_input = weights_ih * input + bias_h.replicate(1, input.cols());
        Eigen::MatrixXd hidden_output = sigmoid(hidden_input);

        Eigen::MatrixXd output_input = weights_ho * hidden_output + bias_o.replicate(1, hidden_output.cols());
        Eigen::MatrixXd output = sigmoid(output_input);

        // 计算输出层的误差
        Eigen::MatrixXd output_error = target - output;
        // 计算输出层的梯度
        Eigen::MatrixXd output_gradient = output_error.array() * sigmoid_derivative(output).array();
        output_gradient *= learning_rate;

        // 计算隐藏层的误差
        Eigen::MatrixXd hidden_error = weights_ho.transpose() * output_gradient;
        // 计算隐藏层的梯度
        Eigen::MatrixXd hidden_gradient = hidden_error.array() * sigmoid_derivative(hidden_output).array();
        hidden_gradient *= learning_rate;

        // 更新权重和偏置
        weights_ho += output_gradient * hidden_output.transpose();
        bias_o += output_gradient.rowwise().sum();

        weights_ih += hidden_gradient * input.transpose();
        bias_h += hidden_gradient.rowwise().sum();
    }
};

int main() {
    // 定义网络结构
    int input_size = 2;
    int hidden_size = 3;
    int output_size = 1;

    // 创建神经网络对象
    NeuralNetwork nn(input_size, hidden_size, output_size);

    // 随机生成训练数据
    Eigen::MatrixXd input = Eigen::MatrixXd::Random(input_size, 10);
    Eigen::MatrixXd target = Eigen::MatrixXd::Random(output_size, 10);

    // 训练网络
    double learning_rate = 0.1;
    int epochs = 1000;
    for (int i = 0; i < epochs; ++i) {
        nn.train(input, target, learning_rate);
    }

    // 进行预测
    Eigen::MatrixXd prediction = nn.feedforward(input);
    std::cout << "Prediction:\n" << prediction << std::endl;

    return 0;
}
```
