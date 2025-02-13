#include <iostream>
#include <Eigen/Dense>
#include <cmath>

// 定义 sigmoid 激活函数
// Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& x) {
//     return 1.0 / (1.0 + (-x).array().exp());
// }

// // 定义 sigmoid 激活函数的导数
// Eigen::MatrixXd sigmoid_derivative(const Eigen::MatrixXd& x) {
//     return x.array() * (1 - x.array());
// }

// // 神经网络类
// class NeuralNetwork {
// private:
//     int input_size;
//     int hidden_size;
//     int output_size;
//     Eigen::MatrixXd weights_ih;  // 输入层到隐藏层的权重
//     Eigen::MatrixXd weights_ho;  // 隐藏层到输出层的权重
//     Eigen::VectorXd bias_h;      // 隐藏层的偏置
//     Eigen::VectorXd bias_o;      // 输出层的偏置

// public:
//     // 构造函数，初始化网络参数
//     NeuralNetwork(int input_size, int hidden_size, int output_size)
//         : input_size(input_size), hidden_size(hidden_size), output_size(output_size) {
//         // 随机初始化权重和偏置
//         weights_ih = Eigen::MatrixXd::Random(hidden_size, input_size);
//         weights_ho = Eigen::MatrixXd::Random(output_size, hidden_size);
//         bias_h = Eigen::VectorXd::Random(hidden_size);
//         bias_o = Eigen::VectorXd::Random(output_size);
//     }

//     // 前向传播
//     Eigen::MatrixXd feedforward(const Eigen::MatrixXd& input) {
//         // 计算隐藏层的输入
//         Eigen::MatrixXd hidden_input = weights_ih * input + bias_h.replicate(1, input.cols());
//         // 应用 sigmoid 激活函数
//         Eigen::MatrixXd hidden_output = sigmoid(hidden_input);

//         // 计算输出层的输入
//         Eigen::MatrixXd output_input = weights_ho * hidden_output + bias_o.replicate(1, hidden_output.cols());
//         // 应用 sigmoid 激活函数
//         Eigen::MatrixXd output = sigmoid(output_input);

//         return output;
//     }

//     // 训练函数
//     void train(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target, double learning_rate) {
//         // 前向传播
//         Eigen::MatrixXd hidden_input = weights_ih * input + bias_h.replicate(1, input.cols());
//         Eigen::MatrixXd hidden_output = sigmoid(hidden_input);

//         Eigen::MatrixXd output_input = weights_ho * hidden_output + bias_o.replicate(1, hidden_output.cols());
//         Eigen::MatrixXd output = sigmoid(output_input);

//         // 计算输出层的误差
//         Eigen::MatrixXd output_error = target - output;
//         // 计算输出层的梯度
//         Eigen::MatrixXd output_gradient = output_error.array() * sigmoid_derivative(output).array();
//         output_gradient *= learning_rate;

//         // 计算隐藏层的误差
//         Eigen::MatrixXd hidden_error = weights_ho.transpose() * output_gradient;
//         // 计算隐藏层的梯度
//         Eigen::MatrixXd hidden_gradient = hidden_error.array() * sigmoid_derivative(hidden_output).array();
//         hidden_gradient *= learning_rate;

//         // 更新权重和偏置
//         weights_ho += output_gradient * hidden_output.transpose();
//         bias_o += output_gradient.rowwise().sum();

//         weights_ih += hidden_gradient * input.transpose();
//         bias_h += hidden_gradient.rowwise().sum();
//     }
// };

// int main() {
//     // 定义网络结构
//     int input_size = 2;
//     int hidden_size = 3;
//     int output_size = 1;

//     // 创建神经网络对象
//     NeuralNetwork nn(input_size, hidden_size, output_size);

//     // 随机生成训练数据
//     Eigen::MatrixXd input = Eigen::MatrixXd::Random(input_size, 10);
//     Eigen::MatrixXd target = Eigen::MatrixXd::Random(output_size, 10);

//     // 训练网络
//     double learning_rate = 0.1;
//     int epochs = 1000;
//     for (int i = 0; i < epochs; ++i) {
//         nn.train(input, target, learning_rate);
//     }

//     // 进行预测
//     Eigen::MatrixXd prediction = nn.feedforward(input);
//     std::cout << "Prediction:\n" << prediction << std::endl;

//     return 0;
// }