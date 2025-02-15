#include <iostream>
#include <Eigen/Dense>
#include <iostream>
#include <Eigen/Dense>
#include "Parameter.h"
#include"AdamOptimizer.h"
#include"LinearLayer.h"
#include <iostream>
#include <Eigen/Dense>
#include <cmath>
using namespace std;
// struct Base
// {
//     Base()
//     {

//     }
// };
// struct Derive1
// {
//     Derive1
//     {

//     }
// };
// struct Derive2
// {
//     Derive2{

//     }
// };
// void f(Base a)
// {
//     //获取a的实际类型T
//     T b;
// }
// template<typename Scalar>
// class Module{
//     public:
//     virtual Eigen::Matrix <Scalar,Eigen::Dynamic,Eigen::Dynamic>
//     forward(const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic,Eigen::Dynamic>&input)=0;
//     virtual Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>backward(const Eigen::Matrix <Scalar,Eigen::Dynamic,Eigen::Dynamic>&grad_output)=0;
//    vitrual void update()=0;
//    virtual void zero_grad()=0;
// };
// template<typename Scalar>
// struct  ReLu:public Module<Scarlar>
// {
//     Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dyanmic>forward(const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>&input)override{
//         this->input=input;
//         return input.cwiseMax(0);
//     }
//     Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>backward
//     (const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>&grad_output)override{
//         Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>grad_input=input;
//         grad_input=(input.array()>0).select(grad_output,0);
//         return grad_input;
//     }
//     void update()override{

//     }
//     void zero_grad()override{

//     }
//     Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>input;
// };
// template<typename Scalar>
// class Sigmoid:public Module<Scalar>{
//     public:
//     Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>forward
//     (const Eigen::Matrix<Scalar,Eigen::Dyanmic,Eigen::Dynamic>&input)override
//     {
//         this->input=input;
//         return 1/(1+(-input.array()).exp());
//     }
//     Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>backward
//     (const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>&grad_output)override{
//         Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>sigmoid=forward(input);
//         return grad_output.cwiseProduct(sigmoid.cwiseProduct
//             (Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Ones(sigmoid.rows(),sigmoid.cols())-sigmoid));
//     }
//     void update()override{

//     }
//     void zero_grad()override{

//     }
//     private:
//     Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>input;
// };
// template<typename Scalar,int Rows,int Cols>
// struct Parameter{
//     private:
//     Eigen::Matrix<Scalar,Rows,Cols>value;
//     Eigen::Matrix<Scalar,Rows,Cols>grad;
//     public:
//     Parameter(const Eigen::Matrix<Scalar,Rows,Cols>&init_value)
//     :value(init_value)
//     {
//         grad=Eigen::Matrix<Scalar,Rows,Cols>::Zero(init_value.rows(),init_value.cols());
//     }
//     Eigen::Matrix<Scalar,Rows,Cols>&get_value()
//     {
//         return value;
//     }
//     const Eigen::Matrix<Scalar,Rows,Cols>&get_value()const 
//     {
//         return value;
//     }
//     const Eigen::Matrix<Scalar,Rows,Cols>&get_grad()const 
//     {
//         return grad;
//     }
    
//     void set_grad(const Eigen::Matrix<Scalar,Rows,Cols>&new_grad)
//     {
//         grad=new_grad;
//     }
//     void zero_grad()
//     {
//         grad.setZero();
//     }
// };
// template<typename Scalar,int Rows,int Cols>
// struct AdamOptimzer:Optimzizer<Scalar,Rows,Cols>
// {
//     Scalar learing_rate;
//     Scalar beta1;
//     Scalar beta2;
//     Scalar epsilon;
//     int timestep;
//     Eigen::Matrix<Scalar,Rows,Cols>m;
//     Eigen::Matrix<Scalar,Rows,Cols>v;
//     AdamOptimizer(Parameter<Scalar,Rows,Cols>&p,
//         Scalar lr=0.001,
//         Scalar b1=0.9,Scalar b2=0.999,Scalar eps=1e-8
//     ):Optimizer<Scalar,Rows,Cols>(p),learing_rate(lr),beta1(b1),beta2(b2),
//     epsilon(eps),timestep(0)
//     {
//         m=Eigen::Matrix<Scalar,Rows,Cols>::Zero(p.get_grad().rows(),p.get_grad().cols());
//         v=Eigen::Matrix<Scalar,Rows,Cols>::Zero(p.get_grad().rows(),p.get_grad().cols());

//     }
//     void step()
//     {
//         timestep++;
//         auto &grad=this->parma.get_grad();
//         auto &value=this->param.get_value();
//         m=beta1*m+(1-beta1)*grad;
//         v=beta2*v+(1-beta2)*grad.cwiseProdut(grad);

//         Eigen::Matrix<Salar,Rows,Cols>m_hat=m/(1-std::pow(beta1,timestep));
//         Eigen::Matrix<Scalar,Rows,Cols>v_hat=v/(1-std::pow(beta2,timestep));

//         Eigen::Matrix<Scalar,Rows,Cols>epsilon_matrix(v_hat.rows(),v_hat.cols())
//        for(int i=0;i<v_hat.rows();i++)
//        {
//         for(int j=0;j<v_hat.cols();j++)
//         {
//             epsilon_matrix(i,j)=epsilon;
//         }
//        }
//        value-=learing_rate*m_hat.cwiseQuotient(v_hat.cwiseSqrt()+epsilon_matrix);
//     }
// };
// template<typename Scalar>
// class LinearLayer:public Module<Scalar>
// {
//     public:
//     Parameter<Scalar,Eigen::Dynamic,Eigen::Dynamic>weight;
//     Parameter<Scalar,1,Eigen::Dynamic>bias;
//     AdamOptimizer<Scalar,Eigen::Dynamic,Eigen::Dynamic>weight_optimizer;
//     AdamOptimizer<Scalar,1,Eigen::Dynamic>bias_optimizer;
//     LinearLayer(int in_features,int out_features)
//     :weight(Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Random(in_features,out_features))
//     ,bias(Eigen::Matrix<Scalar,1,Eigen::Dynamic>::Random(1,out_features)),
//     weight_optimzer(weight),
//     bias_optimzer(bias)
//     {

//     }
//     Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>forward
//     (const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>&input)override
//     {
//         this->input=input;
//         return input*weight.get_value()+bias.get_value();
//     }
//     Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>backward
//     (const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>&grad_output)override
//     {
//         weight.set_grad(weight_grad);
//         Eigen::Marix<Scalar,1,Eigen::Dynamic>bias_grad=grad_output;
//         bias.set_grad(bias_grad);
//         Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>grad_input
//         =grad_output*weight.get_value().transpose();
//         return grad_input;
//     }
//     void update()override{
//         weight_optimzer.step();
//         bias_optimizer.step();
//     }

//     void zero_grad()override{
//         weight.zero_grad();
//         bias.zero_grad();
//     }
//     private:
//     Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>input;
// };
// template<typename Scalar>
// struct loss
// {
//     virtual Scalar forward(const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>&prediciton,
//         const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>&target)=0;
//     virtual Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>backward(const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>&prediction,
//     const Eigen::Matrix<Sclar,Eigen::Dynamic,Eigen::Dynamic>&target)=0;
    
// };
// template<typename Scalar>
// struct MSELoss:publci Loss<Scalar>{
//     Scalar forward(const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic&prediction,
//     const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>&target
//     )override{
//         return (prediction-target).array().square().mean();
//     }
//     Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>copy_backward(const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>&prediciton,
//         const Eigen::Matrix<Sclar,Eigen::Dynamic,Eigen::Dynamic>&target)override
//         {
//             return 2*(prediction-target)/prediction.size();
//         }

// };
//定义 sigmoid 激活函数
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
// #include"AdamOptimizer.h"
// #include"Parameter.h"
// #include "LinearLayer.h"
// 定义Parameter类

// 定义Optimizer基类
// template<typename Scalar, int Rows, int Cols>
// class Optimizer {
// protected:
//     Parameter<Scalar, Rows, Cols>& param;
// public:
//     Optimizer(Parameter<Scalar, Rows, Cols>& p) : param(p) {}

//     virtual void step() = 0;
// };

// 定义Adam优化器类


// 定义全连接层类

// 示例使用
// template <typename Scalar,int Rows,int Cols>
// struct  Optimizer
// {
//     Parameter<Scalar,Rows,Cols>&param;
//     Optimizer(Parameter<Scalar,Rows,Cols>&p):param(p)
//     {

//     }
//     virtual void step()=0;
// };
// template<typename Scalar,int Rows,int Cols>
// struct  AdamOptimizer:Optimizer <Scalar,Rows,Cols>{
//     Scalar learning_rate;
//     Scalar beta1;
//     Scalar beta2;
//     Scalar epsilon;
//     int timestep;
//     Eigen::Matrix<Scalar,Rows,Cols>m;
//     Eigen::Matrix<Scalar,Rows,Cols>v;
//     AdamOptimizer(Parameter<Scalar,Rows,Cols>&p,Scalar lr=0.001
//     ,Scalar b1=0.9,Scalar b2=0.999,Scalar eps=1e-8)
//     :Optimizer<Scalar,Rows,Cols>(p),learning_rate(lr),beta1(b1),beta2(b2),epsilon(eps)
//     {
//         m=Eigen::Matrix<Scalar,Rows,Cols>::Zero(p.get_grad().rows(),p.get_grad().cols());
//         v=Eigen::Matrix<Scalar,Rows,Cols>::Zero(p.get_grad().rows(),p.get_grad().cols());

//     }
//     void step() {
//         timestep++;
//         auto& grad = this->param.get_grad();
//         auto& value = this->param.get_value();

//         // 计算一阶矩估计
//         m = beta1 * m + (1 - beta1) * grad;
//         // 计算二阶矩估计
//         v = beta2 * v + (1 - beta2) * grad.cwiseProduct(grad);

//         // 修正一阶矩估计的偏差
//         Eigen::Matrix<Scalar, Rows, Cols> m_hat = m / (1 - std::pow(beta1, timestep));
//         // 修正二阶矩估计的偏差
//         Eigen::Matrix<Scalar, Rows, Cols> v_hat = v / (1 - std::pow(beta2, timestep));

//         // 将epsilon转换为矩阵
//         //Eigen::Matrix<Scalar, Rows, Cols> epsilon_matrix = Eigen::Matrix<Scalar, Rows, Cols>::Constant(epsilon);
//         Eigen::Matrix<Scalar,Rows,Cols>epsilon_matrix(v_hat.rows(),v_hat.cols());
//         for(int i=0;i<v_hat.rows();i++)
//         {
//             for(int j=0;j<v_hat.cols();j++)
//             {
//                 epsilon_matrix(i,j)=epsilon;
//             }
//         }
//         value -= learning_rate * m_hat.cwiseQuotient(v_hat.cwiseSqrt() + epsilon_matrix);
//     }
// };
// signed main()
// {
//     int in_features=3;
//     int out_features=2;
//     auto t=Eigen::Matrix<double,-1,-1>::Random(3,3);
//     Parameter<double,-1,-1>tt(t);
//     AdamOptimizer<double,Eigen::Dynamic,Eigen::Dynamic>ttt(tt);
// }
// template<typename Scalar>
// class LinearLayer {
// private:
//     Parameter<Scalar, Eigen::Dynamic, Eigen::Dynamic> weight;
//     Parameter<Scalar, 1, Eigen::Dynamic> bias;
//     AdamOptimizer<Scalar, Eigen::Dynamic, Eigen::Dynamic> weight_optimizer;
//     AdamOptimizer<Scalar, 1, Eigen::Dynamic> bias_optimizer;

// public:
//     LinearLayer(int in_features, int out_features)
//         : weight(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Random(in_features, out_features)),
//           bias(Eigen::Matrix<Scalar, 1, Eigen::Dynamic>::Random(1, out_features)),
//           weight_optimizer(weight),
//           bias_optimizer(bias) {
//           }

//     // 前向传播
//     Eigen::Matrix<Scalar, 1, Eigen::Dynamic> forward(const Eigen::Matrix<Scalar, 1, Eigen::Dynamic>& input) {
//         return input * weight.get_value() + bias.get_value();
//     }

//     // 反向传播
//     Eigen::Matrix<Scalar, 1, Eigen::Dynamic> backward(const Eigen::Matrix<Scalar, 1, Eigen::Dynamic>& input,
//                                                       const Eigen::Matrix<Scalar, 1, Eigen::Dynamic>& grad_output) {
//         // 计算权重的梯度
//         Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> weight_grad = input.transpose() * grad_output;
//         weight.set_grad(weight_grad);

//         // 计算偏置的梯度
//         Eigen::Matrix<Scalar, 1, Eigen::Dynamic> bias_grad = grad_output;
//         bias.set_grad(bias_grad);

//         // 计算输入的梯度
//         Eigen::Matrix<Scalar, 1, Eigen::Dynamic> grad_input = grad_output * weight.get_value().transpose();

//         return grad_input;
//     }

//     // 更新参数
//     void update() {
//          weight_optimizer.step();
//          bias_optimizer.step();
//     }

//     // 清零梯度
//     void zero_grad() {
//         weight.zero_grad();
//         bias.zero_grad();
//     }
// };
int main() {
    // 定义输入特征数和输出特征数
    int in_features = 3;
    int out_features = 2;
    // Eigen::Matrix<double,3,3>b;
    auto t=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>::Random(3,3);
    std::cout<<t<<"\n";
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>s;
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>ss=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>::Random(in_features,out_features);
    std::cout<<s<<"\n";

    // std::cout<<b;
    // std::cout<<b.rows();
    //被复制了以后，变成了一个静态的矩阵。
    //
   // 
      Parameter<double,Eigen::Dynamic,Eigen::Dynamic>tt(ss);
      //node<double,-1,-1>w(tt);
    AdamOptimizer<double,Eigen::Dynamic,Eigen::Dynamic>w(tt);
    //std::cout<<"Success\n";


    //创建全连接层对象
     LinearLayer<double> linear_layer(in_features, out_features);

    //模拟输入
    Eigen::Matrix<double, 1, 3> input;
    input << 1.0, 2.0, 3.0;

    // 前向传播
    Eigen::Matrix<double, 1, 2> output = linear_layer.forward(input);
     std::cout << "Output after forward propagation:" << std::endl;
     std::cout << output << std::endl;

    // // 模拟输出的梯度
     Eigen::Matrix<double, 1, 2> grad_output;
     grad_output << 0.1, 0.2;

    // // 反向传播
     linear_layer.zero_grad();
    Eigen::Matrix<double, 1, 3> grad_input = linear_layer.backward(input, grad_output);
     std::cout << "Gradient of input after backward propagation:" << std::endl;
    std::cout << grad_input << std::endl;

    // // 更新参数
     linear_layer.update();

    // // 再次前向传播
     output = linear_layer.forward(input);
     std::cout << "Output after parameter update:" << std::endl;
     std::cout << output << std::endl;

    return 0;
}
// 假设这里定义了 EIGEN_STATIC_ASSERT_FIXED_SIZE 宏
// #define EIGEN_STATIC_ASSERT(CONDITION,MSG) static_assert(CONDITION,#MSG);
// #define EIGEN_STATIC_ASSERT_FIXED_SIZE(TYPE) \
//   EIGEN_STATIC_ASSERT(TYPE::SizeAtCompileTime!=Eigen::Dynamic, \
//                       YOU_CALLED_A_FIXED_SIZE_METHOD_ON_A_DYNAMIC_SIZE_MATRIX_OR_VECTOR)

// // 一个只适用于固定大小矩阵的函数
// template<typename MatrixType>
// void fixedSizeFunction() {
//     EIGEN_STATIC_ASSERT_FIXED_SIZE(MatrixType);
//     std::cout << "This function is called on a fixed-size matrix." << std::endl;
// }

// int main() {
//     // 固定大小的矩阵类型
//     using FixedMatrix = Eigen::Matrix<double, 3, 3>;
//     // 动态大小的矩阵类型
//     using DynamicMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

//     // 调用固定大小函数，传入固定大小矩阵类型
//     fixedSizeFunction<FixedMatrix>();

//     // 调用固定大小函数，传入动态大小矩阵类型，会触发静态断言错误
//      fixedSizeFunction<DynamicMatrix>();

//     return 0;
// }
// #include <iostream>
// #include <Eigen/Dense>
// using namespace std;
// 定义Parameter类

// 定义Optimizer基类
// template<typename Scalar, int Rows, int Cols>
// class Optimizer {
// protected:
//     Parameter<Scalar, Rows, Cols>& param;
// public:
//     Optimizer(Parameter<Scalar, Rows, Cols>& p) : param(p) {}
//     virtual void step() = 0;
// };

// //定义Adam优化器类
// template<typename Scalar, int Rows, int Cols>
// class AdamOptimizer : public Optimizer<Scalar, Rows, Cols> {
// private:
//     Scalar learning_rate;
//     Scalar beta1;
//     Scalar beta2;
//     Scalar epsilon;
//     int timestep;
//     Eigen::Matrix<Scalar, Rows, Cols> m;
//     Eigen::Matrix<Scalar, Rows, Cols> v;

// public:
//     AdamOptimizer(Parameter<Scalar, Rows, Cols>& p, Scalar lr = 0.001,
//                   Scalar b1 = 0.9, Scalar b2 = 0.999, Scalar eps = 1e-8)
//         : Optimizer<Scalar, Rows, Cols>(p), learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), timestep(0) {
//         m = Eigen::Matrix<Scalar, Rows, Cols>::Zero();
//         v = Eigen::Matrix<Scalar, Rows, Cols>::Zero();
//     }

//     void step() override {
//         timestep++;
//         auto& grad = this->param.get_grad();
//         auto& value = this->param.get_value();

//         // 计算一阶矩估计
//         m = beta1 * m + (1 - beta1) * grad;
//         // 计算二阶矩估计
//         v = beta2 * v + (1 - beta2) * grad.cwiseProduct(grad);

//         // 修正一阶矩估计的偏差
//         Eigen::Matrix<Scalar, Rows, Cols> m_hat = m / (1 - std::pow(beta1, timestep));
//         // 修正二阶矩估计的偏差
//         Eigen::Matrix<Scalar, Rows, Cols> v_hat = v / (1 - std::pow(beta2, timestep));

//         // 更新参数
//         value -= learning_rate * m_hat.cwiseQuotient(v_hat.cwiseSqrt() + epsilon);
//     }
// };

// // 定义全连接层类
// template<typename Scalar>
// class LinearLayer {
// private:
//     Parameter<Scalar, Eigen::Dynamic, Eigen::Dynamic> weight;
//     Parameter<Scalar, 1, Eigen::Dynamic> bias;
//     AdamOptimizer<Scalar, Eigen::Dynamic, Eigen::Dynamic> weight_optimizer;
//     AdamOptimizer<Scalar, 1, Eigen::Dynamic> bias_optimizer;

// public:
//     LinearLayer(int in_features, int out_features)
//         : weight(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Random(in_features, out_features)),
//           bias(Eigen::Matrix<Scalar, 1, Eigen::Dynamic>::Random(1, out_features)),
//           weight_optimizer(weight),
//           bias_optimizer(bias) {}

//     // 前向传播
//     Eigen::Matrix<Scalar, 1, Eigen::Dynamic> forward(const Eigen::Matrix<Scalar, 1, Eigen::Dynamic>& input) {
//         return input * weight.get_value() + bias.get_value();
//     }

//     // 反向传播
//     Eigen::Matrix<Scalar, 1, Eigen::Dynamic> backward(const Eigen::Matrix<Scalar, 1, Eigen::Dynamic>& input,
//                                                       const Eigen::Matrix<Scalar, 1, Eigen::Dynamic>& grad_output) {
//         // 计算权重的梯度
//         Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> weight_grad = input.transpose() * grad_output;
//         weight.set_grad(weight_grad);

//         // 计算偏置的梯度
//         Eigen::Matrix<Scalar, 1, Eigen::Dynamic> bias_grad = grad_output;
//         bias.set_grad(bias_grad);

//         // 计算输入的梯度
//         Eigen::Matrix<Scalar, 1, Eigen::Dynamic> grad_input = grad_output * weight.get_value().transpose();

//         return grad_input;
//     }

//     // 更新参数
//     void update() {
//         weight_optimizer.step();
//         bias_optimizer.step();
//     }

//     // 清零梯度
//     void zero_grad() {
//         weight.zero_grad();
//         bias.zero_grad();
//     }
// };

// 示例使用
// signed  main() {
//     // 定义输入特征数和输出特征数
//     int in_features = 3;
//     int out_features = 2;

//     // 创建全连接层对象
//     LinearLayer<double> linear_layer(in_features, out_features);

//     // 模拟输入
//     Eigen::Matrix<double, 1, 3> input;
//     input << 1.0, 2.0, 3.0;

//     // 前向传播
//     Eigen::Matrix<double, 1, 2> output = linear_layer.forward(input);
//     std::cout << "Output after forward propagation:" << std::endl;
//     std::cout << output << std::endl;

//     // 模拟输出的梯度
//     Eigen::Matrix<double, 1, 2> grad_output;
//     grad_output << 0.1, 0.2;

//     // 反向传播
//     linear_layer.zero_grad();
//     Eigen::Matrix<double, 1, 3> grad_input = linear_layer.backward(input, grad_output);
//     std::cout << "Gradient of input after backward propagation:" << std::endl;
//     std::cout << grad_input << std::endl;

//     // 更新参数
//     linear_layer.update();

//     // 再次前向传播
//     output = linear_layer.forward(input);
//     std::cout << "Output after parameter update:" << std::endl;
//     std::cout << output << std::endl;

//     return 0;
// } 
// template<typename Scalar,int Rows,int Cols>
// class Parameter
// {
//     private:
//     Eigen::Matrix<Scalar,Rows,Cols>value;
//     Eigen::Matrix<Scalar,Rows,Cols>grad;

//     public:
//     Parameter(const Eigen::Matrix<Scalar,Rows,Cols>&init_value):value(init_value),grad(Eigen::Matrix<Scalar,Rows,Cols>::Zero())
//     {

//     }
//     const Eigen::Matrix<Scalar,Rows,Cols>&get_value()const 
//      {
//         return value;
//      }
//      cosnt Eigen::Matrix<Scalar,Rows,Cols>&get_grad()const{
//         return grad;
//      }
//      void set_grad(const Eigen::Matrix<Scalar,Rows,Cols>&new_grad){
//         grad=new_grad;
//      }
//      void zero_grad()
//      {
//         grad.setZero();
//      }
// };
// template<typename Scalar,int Rows,int Cols>
// class Optimizer{
//     protected:
//         Parameter<Scalar,Rows,Cols>&param;
//     public:
//     Optimizer(Parameter<Scalar,Rows,Cols>&p):param(p){

//     }
//     virtual void step()=0;
// };
// template<typename Scalar, int Rows,int Cols>
// class AdamOptimizer:public Optimizer<Scalar,Rows,Cols>{
//     private:
//     Scalar learing_rate;
//     Scalar beta1;
//     Scalar beta2;
//     Scalar epsilon;
//     int timestep;
//     Eigen::Matrix<Scalar,Rows,Cols>m;
//     Eigen::Matrix<Scalar,Rows,Cols>v;
//     public:
//     AdamOptimizer(Parameter<Scalar,Rows,Cols>&p,Scalar lr=0.001,Scalar b1=0.9,Scalar b2=0.999,Scalar eps=1e-8):
//     Optimizer<Scalar,Rows,Cols>(p),learing_rate(lr),beta1(b1),beta2(b2),epsilon(eps),timestep(0)
//     {
//         m=Eigen::Matrix<Scalar,Rows,Cols>::zero();
//         v=Eigen::Matrix<Scalar,Rows,Cols>::zero();
//     }
//     void step()override{
//         timestep++;
//         auto &grad=this->param.get_grad();
//         auto &value=this->param.get_value();;
//         m=beta1*m+(1-beta1)*grad;
//         v=beta2*v+(1-beta2)*grad.cwiseProduct(grad);

//         Eigen::Matrix<Scalar,Rows,Cols>m_hat=m/(1-std::pow(beta1,timestep));
//         Eigen::Matrix<Scalar,Rows,Cols>v_hat=v/(1-std::pow(beta2,timestep));

//         value-=learing_rate*m_hat.cwiseQuotient(v_hat.cwiseSqrt()+epsilon);
//     }
// };
// template<typename Scalar>
// class LinearLayer{
//     private:
//     Parameter<Scalar,Eigen::Dynamic,Eigen::Dynamic>weight;
//     Parameter<Scalar,1,Eigen::Dynamic>bias;
//     AdamOptimizer<Scalar,Eigen::Dynamic,Eigen::Dynamic<>weight_optimizer;

//     public:
//     LinearLayer(int in_features,int out_features)
//     :weight(Eigne::Matrix<Scalar>,Eigen::Dynamic,Eigen::Dynamic>::Random(in_features,out_features)),
//     bias(Eigen::Matrix<Scalar,1,Eigen::Dynamic>::Random(1,out_features)),
//     weight_optimizer(weight),
//     bias_optimizer(bias)
//     {

//     }
//     Eigen::Matrix<Scalar,1,Eigen::Dynamic>forward(const Eigen::Matrix<Sclar,1,Eigen::Dynamic&input)
//     {
//         return input*weight.get_value()+bias.get_value();
//     }
//     Eigen::Matrix<Scalar,1,Eigen::Dynamic>backward(const Eigen::Matrix<Scalar,1,Eigen::Dynamic>&input,
//         const Eigen::Matrix<Scalar,1,Eigen::Dynamic>&grad_output)
//     {
//         Eigen::Matrix<Scaar,Eigen::Dynamic,Eigen::Dynamic>weight_grad=input.transpose()*grad_output;
//         weight.set_grad(weight_grad);

//         Eigen::Matrix<Scalar,1,Eigen::Dynamic>bias_grad=grad_output;
//         bias.set_grad(bias_grad);
//         Eigen::Matrix<Scalar,1,Eigen::Dynamic>grad_input=grad_output*weight.get_value().transpose();
//         return grad_input;
//     }
//     void update()
//     {
//         weight_optimizer.step();
//         bias_optimizer.step();
//     }
//     void zero_grad()
//     {
//         weight.zero_grad();
//         bias.zero_grad();
//     }
// };
// signed main()
// {
//     int in_features=3;
//     int out_feature=2;
//     LinearLayer<double>linear_layer(in_features,out_feature);

//     Eigen::Matrix<double,1,3>input;
//     input<<1.0,2.0,3.0;

//     Eigen::Matrix<double,1,2>output=linear_layer.fowward(input);
//     Eigen::Matrix<double,1,2>grad_output;
//     grad_output<<0.1,0.2;

//     linear_layzer.zero_gard();
//     Eigen::Matrix<double,1,3>grad_input=linear_layer.backward(input,grad_output);
//     linear_layer.update();
//     output=linear_layer.forward(input);
// }
// const int mod=1e9+7;
// template <int mod>
// struct MInt{
//     int val;
//     int normalize(int x)const{
//         x%=mod;
//         if(x<0)x+=mod;
//         return x;
//     }
//     MInt():val(0){

//     }
//     MInt(int x):val(normalize(x)){}
//     MInt operator+(const MInt&other)const 
//     {
//         return MInt(val+other.val);
//     }
//     MInt operator-(const MInt&other)const 
//     {
//         return MInt(val-other.val);
//     }
//     MInt operator*(const MInt&other)const 
//     {
//         return MInt(1LL*val*other.val%mod);
//     }
//     MInt operator/(const MInt&other)const{
//         if(other.val==0)
//         {
//             throw std::runtime_error("Division by zero");
//         }
//         int inv=1;

//         int b=other.val;
//         int m=mod-2;
//         while(m>0)
//         {
//             if(m&1)
//             {
//                 inv=1LL*inv*b%mod;
//             }
//             b=1LL*b*b%mod;
//             m>>=1;
//         }
//         return MInt(1LL*val*inv%mod);
//     }
//     MInt&operator+=(const MInt&other)
//     {
//         val=normalize(val+other.val);
//         return *this;
//     }
//     MInt&operator-=(const MInt&other)
//     {
//         val=normalize(val-other.val);
//         return *this;
//     }
//     MInt&operator*=(const MInt&other)
//     {
//         val=1LL*val*other.val%mod;
//         return *this;
//     }
//     MInt& operator/=(const MInt&other)
//     {
//         *this=*this/other;
//         return *this;
//     }
//     bool operator==(const MInt&other)const 
//     {
//         return val==other.val;
//     }
//     bool operator!=(const MInt&other)const 
//     {
//         return val!=other.val;
//     }
//     int value()const 
//     {
//         return val;
//     }
// };
// MInt<mod>f[1010][1010][2];
// bool vis[1010][1010][2];
// int over;
// MInt<mod>p1,up1;
// MInt<mod>p2,up2;

// MInt<mod>dfs(int x,int y,int player)
// {
//     if(x==over)return 0;
//     if(y==over)return 1;
//     if(vis[x][y][player])
//     {
//         return f[x][y][player];
//     }
//     vis[x][y][player]=1;
//     MInt<mod>res;
//     if(player==0)
//     {
//         res=dfs(x+1,y,1)*p1+dfs(x,y+1,0)*p2*(up1);
//         MInt<mod>de=MInt<mod>(1)-up1*up2;
//         res/=de;
//     }
//     else 
//     {
//         res=dfs(x,y+1,0)*p2+dfs(x,y,0)*up2;

//     }
//     return f[x][y][player]=res;
// }
// signed main()
// {
//     cin>>over;
//     int a,b;
//     cin>>a>>b;
//     p1=MInt<mod>(a)/MInt<mod>(b);
//     up1=MInt<mod>(1)-p1;

//     cin>>a>>b;
//     p2=MInt<mod>(a)/MInt<mod>(b);
//     up2=MInt<mod>(1)-p2;

//     MInt<mod>res=dfs(0,0,0);
//     cout<<res.val;
// }
// const int mod=1e9+7;
// template <int mod>
// struct  MInt {

//     int val;

//     // 确保值在 [0, mod) 范围内
//     int normalize(int x) const {
//         x %= mod;
//         if (x < 0) x += mod;
//         return x;
//     }
//     // 默认构造函数
//     MInt() : val(0) {}

//     // 从整数构造
//     MInt(int x) : val(normalize(x)) {}

//     // 重载加法运算符
//     MInt operator+(const MInt& other) const {
//         return MInt(val + other.val);
//     }

//     // 重载减法运算符
//     MInt operator-(const MInt& other) const {
//         return MInt(val - other.val);
//     }

//     // 重载乘法运算符
//     MInt operator*(const MInt& other) const {
//         return MInt(1LL * val * other.val % mod);
//     }

//     // 重载除法运算符
//     MInt operator/(const MInt& other) const {
//         if (other.val == 0) {
//             throw std::runtime_error("Division by zero in modular arithmetic");
//         }
//         // 计算逆元
//         int inv = 1;
//         int b = other.val;
//         int m = mod - 2;
//         while (m > 0) {
//             if (m & 1) {
//                 inv = 1LL * inv * b % mod;
//             }
//             b = 1LL * b * b % mod;
//             m >>= 1;
//         }
//         return MInt(1LL * val * inv % mod);
//     }

//     // 重载 += 运算符
//     MInt& operator+=(const MInt& other) {
//         val = normalize(val + other.val);
//         return *this;
//     }

//     // 重载 -= 运算符
//     MInt& operator-=(const MInt& other) {
//         val = normalize(val - other.val);
//         return *this;
//     }

//     // 重载 *= 运算符
//     MInt& operator*=(const MInt& other) {
//         val = 1LL * val * other.val % mod;
//         return *this;
//     }

//     // 重载 /= 运算符
//     MInt& operator/=(const MInt& other) {
//         *this = *this / other;
//         return *this;
//     }

//     // 重载 == 运算符
//     bool operator==(const MInt& other) const {
//         return val == other.val;
//     }

//     // 重载 != 运算符
//     bool operator!=(const MInt& other) const {
//         return val != other.val;
//     }
    
//     // 获取值
//     int value() const {
//         return val;
//     }
// };
// //f[i][j][0]表示的是当前局面(i,j)，先手出，先手获胜的概率。
// //f[i][j][1]表示当前局面时(i,j),后手出，先手获胜的概率。
// //f[i][j][0]=(f[i+1][j][1])*(p_1)+(f[i][j][1])*((1-P1))
// //f[i+1][j][1]=(f[i][j+1][1])*P2+f[i][j][0]*(1-p2);
// //f[i][j][0]=(f[i+1][j][1])*(P_1)+(f[i][j+1][0]*P2+f[i][j][0]*(1-p2))*((1-P1))
// //f[i][j][0]-(1-p2)*(1-p1)*(f[i][j][0])=f[i+1][j][1]*p1+f[i][j+1][1]*p2*(1-p1)
// //f[i][j]
// //f[i][j][1]=(f[i][j+1][0])*P2+(f[i][j][0])*UP2;
// MInt<mod> f[1010][1010][2];
// bool vis[1010][1010][2];
// int over;
// MInt<mod> p1,up1;
// MInt<mod> p2,up2;
// MInt<mod> dfs(int x,int y,int player)
// {
//     if(x==over)return 0;
//     if(y==over)return 1;
//     if(vis[x][y][player])
//     return f[x][y][player];
//     vis[x][y][player]=1;
//     MInt<mod>res;
//     if(player==0)
//     {
//         res=dfs(x+1,y,1)*p1+dfs(x,y+1,0)*p2*(up1);
//         MInt<mod>de=MInt<mod>(1)-up1*up2;
//         res/=de;
//     }
//     else{
//         res=dfs(x,y+1,0)*p2+dfs(x,y,0)*up2;
//     }
//     return f[x][y][player]=res;
// }
// signed main()
// {
//     cin>>over;
//     int a,b;
//     cin>>a>>b;
//     p1=MInt<mod>(a)/MInt<mod>(b);
//     up1=MInt<mod>(1)-p1;
//     cin>>a>>b;
//     p2=MInt<mod>(a)/MInt<mod>(b);
//     up2=MInt<mod>(1)-p2;
//     MInt<mod>res=dfs(0,0,0);
//     cout<<res.val;
// }
// const int N=5010;
// const int dirs[4][2]={{-1,0},{1,0},{0,-1},{0,1}};
// int n;
// int values[N][N];
// bool vis[N][N];
// bool inR(int x,int y)
// {
//     return x>=0&&x<n&&y>=0&&y<n;
// }
// array<int,3>bfs(const vector<string>&G,const int x,const int y)
// {
//     queue<pair<int,int>>q;
//     q.push({x,y});
//     vis[x][y]=true;

//     int siz=0;
//     set<pair<int,int>>blk;

//     while(!q.empty())
//     {
//         auto [cx,cy]=q.front();
//         q.pop();
//         siz++;
//         for(const auto &dir:dirs)
//         {
//             int nx=cx+dir[0],ny=cy+dir[1];
//             if(inR(nx,ny))
//             {
//                 if(G[nx][ny]=='*'&&!vis[nx][ny])
//                 {
//                     vis[nx][ny]=true;
//                     q.push({nx,ny});
//                 }
//                 else if(G[nx][ny]=='.')
//                 blk.insert({nx,ny});
//             }
//         }
//     }
//     if(blk.size()==1)
//     {
//         pair<int,int>tp;
//         tp=*blk.begin();
//         return {tp.first,tp.second,siz};
//     }
//     else {
//         return {-1,-1,-1};
//     }
// }
// signed main()
// {
//     memset(values,0,sizeof values);
//     memset(vis,0,sizeof vis);

//     cin>>n;
//     vector<string>G(n);

//     for(int i=0;i<n;i++)
//     cin>>G[i];

//     int R=0;
//     for(int i=0;i<n;i++)
//     {
//         for(int j=0;j<n;j++)
//         {
//             if(!vis[i][j]&&G[i][j]=='*')
//             {
//                 auto [x,y,sz]=bfs(G,i,j);
//                 if(x<0)continue;
//                 values[x][y]+=sz;
//                 R=max(values[x][y],R);
//             }
//         }
//     }
//     cout<<R<<"\n";
//     return 0;
// }
// const int N=1010;
// const int mod=998244353;
// int n,m;
// vector<int>a[N*N];
// int pref[N*N];
// int prex[3][N*N];
// int prey[3][N*N];
// int f[N*N];
// int pow_m(int a,int x)
// {
//     int res=1;
//     while(x)
//     {
//         if(x&1)
//         {
//             res=res*a;
//             res%=mod;
//         }
//         a=a*a%mod;
//         x>>=1;
//     }
//     return res;
// }
// int inv(int x)
// {
//     return pow_m(x,mod-2);
// }
// signed main()
// {
//     cin>>n>>m;
//     for(int i=1;i<=n;i++)
//     {
//         for(int j=1;j<=m;j++)
//         {
//             int b;
//             cin>>b;
//             a[(i-1)*m+j]=vector<int>{b,i,j};
//         }
//     }
//     sort(a+1,a+n*m+1);

//     int sx,sy;
//     cin>>sx>>sy;
//     for(int i=1;i<=n*m;i++)
//     {
//         int l=0,r=n*m;
//         while(l<r)
//         {
//             int mid=l+r+1>>1;
//             if(a[mid][0]<a[i][0])
//             l=mid;
//             else r=mid-1;
//         }
//         if(l!=0)
//         {
//             f[i]+=(prex[2][l]+prey[2][l])%mod;
//             f[i]%=mod;

//             f[i]+=l*a[i][2]*mod*a[i][2]%mod;
//             f[i]%=mod;

//             f[i]+=l*a[i][1]%mod*a[i][1]%mod;
//             f[i]%=mod;

//             f[i]=(f[i]-2*a[i][1]%mod*prex[1][l]%mod+2*mod)%mod;
//             f[i]=(f[i]-2*a[i][2]%mod*prey[1][l]%mod+2*mod)%mod;
//             f[i]+=pref[l];
//             f[i]%=mod;
//         }
//         f[i]=f[i]*inv(l)%mod;
//         pref[i]=pref[i-1]+f[i];
//         pref[i]%=mod;
//         prex[2][i]=prex[2][i-1]+a[i][1]*a[i][1]%mod;
//         prex[2][i]%=mod;

//         prey[2][i]=prey[2][i-1]+a[i][2]*a[i][2]%mod;
//         prey[2][i]%=mod;

//         prex[1][i]=prex[1][i-1]+a[i][1];
//         prex[1][i]%=mod;

//         prey[1][i]=prey[1][i-1]+a[i][2];
//         prey[1][i]%=mod;

//         if(a[i][1]==sx&&a[i][2]==sy)
//         {
//             cout<<f[i];
//             return 0;
//         }
//     }
// }
// const int mod=998244353;
// const int N=100010;
// const int M=7*N;
// int n,m;
// int h[N],e[M],ne[M],idx;

// void add(int a,int b){
//     e[idx]=b;
//     ne[idx]=h[a];
//     h[a]=idx++;
// }
// int f[100010];
// int g[100010];
// int s1,s2;
// void dp(int now)
// {
//     if(g[now])return;
//     g[now]=1;

//     for(int i=h[now];~i;i=ne[i])
//     {
//         int j=e[i];
//         dp(j);
//         g[now]+=g[j];
//         g[now]%=mod;
//         f[now]+=f[j]+g[j];
//         f[now]%=mod;
//     }
// }
// int pow_m(int x,int y)
// {
//     int ret=1;
//     x%=mod;
//     while(y)
//     {
//         if(y&1)ret=ret*x%mod;
//         y>>=1;
//         x=x*x%mod;
//     }
//     return ret;
// }
// signed main()
// {
//     memset(h,-1,sizeof h);
//     cin>>n>>m;
//     for(int i=1,x,y;i<=m;i++)
//     {
//         cin>>x>>y;
//         add(x,y);
//     }
//     for(int i=1;i<=n;i++)
//     {
//         if(!g[i])dp(i);
//     }
//     for(int i=1;i<=n;i++)
//     {
//         s1+=f[i];
//         s1%=mod;
//         s2+=g[i];
//         s2%=mod;
//     }
//     cout<<s1*pow_m(s2,mod-2)%mod;
// }
//1 4
//2 5
//3
//1 2 3
//4 5 6
//3
//1 3 4 6 2 5 1 6 2 5 3 4
//1  5 2  4  3 6
//1 2
//2 3
//4 5

//1 6
//2 5
//3 4


//1 3
//2 4
//5 6
//7 8
//9 10

//1 2 
//3 4
//5 6
//7 8
//9 10

// void solve()
// {
//     int n,k;
//     cin>>n>>k;
//     if(k<n)
//     {
//         cout<<-1<<"\n";
//         return ;
//     }
//     int sum=0;
//     for(int i=1;i<=n;i++)
//     {
//         sum+=2*n-i+1-i;
//     }
//     if(sum<k)
//     {
//         cout<<-1<<"\n";
//         return ;
//     }
//     if((k-n)%2!=0)
//     {
//         cout<<-1<<"\n";
//         return ;
//     }
//     vector<int>a(n*2+100);
//     for(int i=1;i<=n*2;i++)
//     a[i]=i;
//     while(1)
//     {
//         int sum=0;
//         for(int i=1;i<=2*n;i+=2)
//         {
//             sum+=a[i+1]-a[i];
//         }
//         if(sum==k)
//         {
//             for(int i=1;i<=n;i++)cout<<a[i*2-1]<<" "<<a[i*2]<<"\n";
//             return;
//         }
//         swap(a[rand()%n+1],a[rand()%n+1]);

//     }
// }
// signed main()
// {
//     // int n=8;
//     // vector<int>a(n+100);
//     // for(int i=1;i<=n;i++)
//     // {
//     //     a[i]=i;
//     // }
//     // set<int>vis;
//     // do{
//     //     int sum=0;
//     //     for(int i=1;i<=n;i+=2)
//     //     sum+=abs(a[i+1]-a[i]);
//     //     if(!vis.count(sum))
//     //     {
//     //         vis.insert(sum);
//     //         cout<<sum<<" ";
//     //         for(int i=1;i<=n;i++)
//     //         {
//     //             cout<<a[i]<<" ";
//     //         }
//     //         cout<<"\n";
//     //     }
        
//     // }
//     //while(next_permutation(a.begin()+1,a.begin()+n+1));
//     int T;
//     cin>>T;
//     while(T--)solve();
// }
// Eignen::matrixXd sigmoid(const Eigen::MatrixXd&x)
// {
//     return 1.0/(1.0+(-x).array().exp);
// }
// Eigen::MatrixXd sigmoid_derivative(const Eigen::MatrixXd&x)
// {
//     return x.array()*(1-x.array());
// }
// struct  nn{
//   int input_size;
//   int hidden_size;
//   int output_size;
//   Eigen::MatrixXd weights_ih;
//   Eigen::MatrixXd weights_ho;
//   Eigen::VectorXd bias_h;
//   Eigen::VectorXd bias_o;
//     nn(int input_size,int hidden_size,int output_size):input_size(input_size),hidden_size(hidden_size),output_size(output_size)
//     {
//         weights_ih=Eligen::MatrixXd::Random(hidden_size,input_size);
//         weights_ho=Eligen::MatrixXd::Random(output_size,hidden_size);

//         bias_h=Eigen::VectorXd::Random(hidden_size);

//         bias_o=Eigen::VectorXd::Random(output_size);
//     }
//     Eigen::MatrixXd feedforward(const Eigen::MatrixXd &input)
//     {
//         Eigen::MatrixXd hidden_input=weights_ih*input+bias_h.replicate(1,inputs.cols());
//         Eigen::MatrixXd hideen_output=sigmoid(hidden_input);
//         Eigen::MatrixXd output_input=weights_ho*hidden_output+bias_o.replicate(1,hidden_output);

//         Eigen::MatrixXd output=sigmoid(output_input);
//         return output;
//     }
//     void train(const Eigen::MatrixXd&input,const Eigen::MatrixXd&target,double learing_rate)
//     {
//         Eigen::MatrixXd hidden_input=weighs_ih*input+bias_h.replicate(1,inputs.cols());
//         Eigen::Matrix hidden_output=sigomid(hidden_input);

//         Eigen::MatrixXd output_input=weights_ho*hiddent_output+bias_o.replicate(1,hidden_output);

//         Eigen:MatrixXd output=sigmoid(output_input);

//         Eigen::Matrix output_error=target-output;

//         Eigen::MatrixXd output_gradient=output_error.array()*sigmoid_derviative(putput).array();
//         output_gradient*=learing_rate;
//         Eigen::MatrixXd hidden_error=weights_ho.transpose()*output_gradient;
//         Eigen::MatrixXd hidden_gradient=hidden_error.array()*sigmoid_derivative(hidden_output).array();
//         hidden_gradient*=learing_rate;

//         weights_ho+=output_gradient*hidden_output.transpose();

//         bias_o+=output_gradient.rowwise().sum();
//         weights_ih+=hidden_gradient*input.transpose();
//         bias_h+=hidden_gradinet.roweise().sum();
//     }
// };
// signed main()
// {
//     int input_size=2;
//     int hidden_size=3;
//     int output_size=1;

//     nn w(input_size,hidden_size,output_size);
//     Eigen::MatrixXd input=Eigen::MatrixXd::Random(input_size,10);
//     Eigen::MatrixXd target=Eigen::MatrixXd::Random(output_size,10);
//     double learing_rate=0.1;

//     int epochs=1000;
//     for(int i=1;i<epochs;i++)
//     {
//         w.train(input,target,learing_rate);
//     }
//     Eigen::MatrixXd prediction=w.feedforward(input);
//     cout<<"predicion:\n"<<prediction<<"\n                        ";
// }
// template<int MOD>
// int mod_inverse(int a)
// {
//     int m=MOD,x=1,y=0;
//     while(a>1)
//     {
//         int q=a/m;
//         int t=m;
//         m=a%m,a=t;
//         t=y;
//         y=x-q*y;
//         x=t;
//     }
//     return x<0?x+MOD:x;
// }
// template<int MOD>
// struct MInt{
//     int x;
//     MInt():x(0){}
//     MInt(long long y):x(y>=0?y%MOD:(MOD-(-y)%MOD)%MOD)
//     {

//     }
//     MInt operator+(const MInt&other)const 
//     {
//         return MInt(x+other.x);
//     }
//     MInt operator-(const MInt &other)const 
//     {
//         return MInt(x-other.x);
//     }
//     MInt operator*(const MInt&other)const 
//     {
//         return MInt(1LL*x*other.x);
//     }
//     MInt&operator++()
//     {
//         x=(x+1)%MOD;
//         return *this;
//     }
//     MInt&operator+=(const MInt&other)
//     {
//         x=(x+other.x)%MOD;
//         return *this;
//     }
//     MInt &operator*=(const MInt&other)
//     {
//         x=1LL*(x*other.x)%MOD;
//         return *this;
//     }
//     MInt&operator/=(const MInt&other)
//     {
//         x=1LL*x*mod_inverse<MOD>(other.x)%MOD;
//         return *this;
//     }
//     friend ostream&operator<<(ostream&os,const MInt&m)
//     {
//         os<<m.x;
//         return os;
//     }
// };  
// const int mod=998244353;
// typedef MInt<mod>Z;
// Z binom(int m,int k)
// {
//     if(m<k||k<0)return 0;
//     Z ret=1;
//     for(int i=1;i<=k;i++)
//     {
//         ret*=MInt<mod>(m-i+1);
//     }
//     for(int i=1;i<=k;i++)
//     ret/=i;
//     return ret;
// }
// void solve()
// {
//     int k;
//     int n;
//     cin>>k>>n;
//     int sum=0;
//     vector<vector<Z>>dp(k+1,vector<Z>(20,Z(0)));

//     for(int i=2;i<=k;i++)
//     {
//         dp[i][1]=1;
//         for(int j=1;j<19;j++)
//         {
//             for(int d=2;d*i<=k;d++)
//             {
//                 dp[i*d][j+1]+=dp[i][j];
//             }
//         }
//     }
//     for(int i=1;i<=k;i++)
//     {
//         Z sum=0;
//         for(int j=1;j<=19;j++)
//         {
//             sum+=binom(n+1,j+1)*dp[i][j];
//         }
//         if(i==1)sum=n;
//         cout<<sum<<" ";
//     }
//     cout<<"\n";
// }
// signed main()
// {
//     int T;
//     cin>>T;
//     while(T--)solve();
// }
// void solve()
// {
//     int n;
//     cin>>n;
//     int A[n];
//     for(int &i:A)
//     {
//         cin>>i;
//     }
//     int ans=0;
//     for(int k=1;k<=n;k++)
//     {
//         if(n%k==0)
//         {
//             int g=0;
//             for(int i=0;i+k<n;i++)
//             {
//                 g=__gcd(g,abs(A[i+k]-A[i]));
//             }
//             ans+=(g!=1);
//         }
//     }
//     cout<<ans<<'\n';
// }
// signed main()
// {
//     int T;
//     cin>>T;
//     while(T--)solve();
// }
// template <int MOD>
// int mod_inverse(int a)
// {
//     int m=MOD,x=1,y=0;
//     while(a>1)
//     {
//         int q=a/m;
//         int t=m;
//         m=a%m,a=t;
//         t=y;
//         y=x-q*y;
//         x=t;
//     }
//     return x<0?x+MOD:x;
// }
// template<int MOD>
// struct  MInt{
//     int x;
//     MInt():x(0){}

//     MInt(long long y):x(y>=0?y%MOD:(MOD-(-y)%MOD)%MOD){}
//     MInt operator+(const MInt&other)const 
//     {
//         return MInt(x+other.x);
//     }
//     MInt operator-(const MInt&other)const 
//     {
//         return MInt(x-other.x);
//     }
//     MInt operator*(const MInt&other)const{
//         return MInt(1LL*x*other.x);
//     }

//     MInt&operator++()
//     {
//         x=(x+1)%MOD;
//         return *this;
//     }
//     MInt&operator+=(const MInt&other)
//     {
//         x=(x+other.x)%MOD;
//         return *this;
//     }
//     MInt&operator*=(const MInt&other)
//     {
//         x=1LL*(x*other.x)%MOD;
//         return *this;
//     }
//     MInt&operator/=(const MInt&other)
//     {
//         x=1LL*x*mod_inverse<MOD>(other.x)%MOD;
//         return *this;
//     }
//     friend ostream&operator<<(ostream&os,const MInt&m)
//     {
//         os<<m.x;
//         return os;
//     }

// };
// const int mod=998244353;
// typedef MInt<mod> Z;
// Z binom(int m,int k)
// {
//     if(m<k||k<0)return 0;
//     Z ret=1;
//     for(int i=1;i<=k;i++)
//     ret*=MInt<mod>(m-i+1);
//     for(int i=1;i<=k;i++)ret/=i;
//     return ret;
// }
// void solve()
// {
//     int k;
//     int n;
//     cin>>k>>n;
//     int sum=0;
//     vector<vector<Z>>dp(k+1,vector<Z>(20,Z(0)));
//     for(int i=2;i<=k;i++)
//     {
//         dp[i][1]=1;
//         for(int j=1;j<19;j++)
//         {
//             for(int d=2;d*i<=k;d++)
//             {
//                 dp[i*d][j+1]+=dp[i][j];
//             }
//         }
//     }
//     for(int i=1;i<=k;i++)
//     {
//         Z sum=0;
//         for(int j=1;j<=19;j++)
//         {
//             sum+=binom(n+1,j+1)*dp[i][j];
//         }
//         if(i==1)sum=n;
//         cout<<sum<<" ";
//     }
//     cout<<"\n";
// }
// signed main()
// {
//     int T;
//     cin>>T;
//     while(T--)solve();
// }#include <iostream>
// #include <Eigen/Dense>
// #include <cmath>
// #include <vector>

// using namespace Eigen;
// using namespace std;
// signed main()
// {
//     Eigen::MatrixXd m(2,2);
//     m(0,0)=3;
//     m(1,0)=2.5;
//     m(0,1)=m(0,0)+m(1,0);

//     cout<<m<<"\n";
//     m<<1,2,3,4;
//     cout<<m<<"\n";
//     VectorXd v(2);
//     v(0)=4;
//     v(1)=v(0)-1;
//     cout<<v<<"\n";
//     m.resize(4,3);

//     cout<<m.rows()<<" "<<m.cols()<<"\n";
//     cout<<m<<"\n";

//     v.resize(5);
//     cout<<v.size()<<"\n";
//     cout<<v.rows()<<"x"<<v.cols()<<"\n";
//     Matrix2d mat;
//     mat<<1,2,3,4;
//     Vector2d u(-1,1);
//     v.resize(2);
//     v(0)=1;
//     v(1)=1;
//     cout<<mat*mat<<"\n";
//     cout<<mat*u<<"\n";
//     cout<<u.transpose()*mat<<"\n";
//     cout<<u.transpose().rows()<<" "<<u.transpose().cols()<<"\n";
//     cout<<v.rows()<<" "<<v.cols()<<"\n";
//     cout<<u.transpose()*v<<"\n";
//     cout<<u*v.transpose()<<"\n";
// }
// 神经网络类
// class NeuralNetwork {
// private:
//     struct Layer {
//         MatrixXd weights;   // 权重矩阵
//         VectorXd bias;      // 偏置向量
//         VectorXd output;    // 输出值
//         VectorXd delta;     // 误差项
        
//         Layer(int input_size, int output_size) :
//             weights(MatrixXd::Random(output_size, input_size) * 0.1),
//             bias(VectorXd::Random(output_size) * 0.1),
//             output(VectorXd::Zero(output_size)),
//             delta(VectorXd::Zero(output_size)) {}
//     };

//     vector<Layer> layers;
//     double learning_rate;

//     // Sigmoid激活函数
//     static VectorXd sigmoid(const VectorXd& x) {
//         return 1.0 / (1.0 + (-x.array()).exp());
//     }

//     // Sigmoid导数
//     static VectorXd sigmoid_derivative(const VectorXd& x) {
//         return x.array() * (1.0 - x.array());
//     }

// public:
//     // 网络结构构造器 (示例: {2, 4, 1} 表示输入层2节点，隐藏层4节点，输出层1节点)
//     NeuralNetwork(const vector<int>& topology, double lr = 0.1) 
//         : learning_rate(lr) 
//     {
//         for (size_t i = 0; i < topology.size() - 1; ++i) {
//             layers.emplace_back(topology[i], topology[i+1]);
//         }
//     }

//     // 前向传播
//     VectorXd forward(const VectorXd& input) {
//         layers[0].output = input;
//         for (size_t i = 1; i < layers.size(); ++i) {
//             layers[i].output = sigmoid(
//                 layers[i].weights * layers[i-1].output + layers[i].bias
//             );
//         }
//         return layers.back().output;
//     }

//     // 反向传播
//     void backward(const VectorXd& target) {
//         // 计算输出层误差
//         Layer& output_layer = layers.back();
//         output_layer.delta = (output_layer.output - target).array() * 
//                              sigmoid_derivative(output_layer.output).array();

//         // 反向传播误差
//         for (int i = layers.size()-2; i >= 0; --i) {
//             layers[i].delta = sigmoid_derivative(layers[i].output).array() * 
//                              (layers[i+1].weights.transpose() * layers[i+1].delta).array();
//         }

//         // 更新权重和偏置
//         for (size_t i = layers.size()-1; i > 0; --i) {
//             layers[i].weights -= learning_rate * layers[i].delta * layers[i-1].output.transpose();
//             layers[i].bias -= learning_rate * layers[i].delta;
//         }
//     }

//     // 训练函数
//     void train(const vector<VectorXd>& inputs, 
//               const vector<VectorXd>& targets, 
//               int epochs) 
//     {
//         for (int epoch = 0; epoch < epochs; ++epoch) {
//             double total_error = 0.0;
//             for (size_t i = 0; i < inputs.size(); ++i) {
//                 VectorXd output = forward(inputs[i]);
//                 backward(targets[i]);
//                 total_error += 0.5 * (output - targets[i]).squaredNorm();
//             }
//             if (epoch % 1000 == 0) {
//                 cout << "Epoch: " << epoch << "\tError: " << total_error << endl;
//             }
//         }
//     }
// };

// // 测试XOR问题
// int main() {
//     // 训练数据
//     vector<VectorXd> inputs = {
//         Vector2d(0, 0),
//         Vector2d(0, 1),
//         Vector2d(1, 0),
//         Vector2d(1, 1)
//     };

//     vector<VectorXd> targets = {
//         VectorXd::Constant(1, 0.0),
//         VectorXd::Constant(1, 1.0),
//         VectorXd::Constant(1, 1.0),
//         VectorXd::Constant(1, 0.0)
//     };

//     // 创建网络 (2-4-1结构)
//     NeuralNetwork nn({2, 4, 1}, 0.5);
    
//     // 训练10000次
//     nn.train(inputs, targets, 10000);

//     // 测试输出
//     cout << "\nTest Results:" << endl;
//     for (const auto& input : inputs) {
//         VectorXd output = nn.forward(input);
//         cout << input.transpose() << " => " 
//              << output(0) << " (rounded: " << round(output(0)) << ")" << endl;
//     }

//     return 0;
// }