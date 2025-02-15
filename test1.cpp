#include <bits/stdc++.h>
#include <Eigen/Dense>
#include "Module.h"
#include "LinearLayer.h"
#include "Parameter.h"
#include "AdamOptimizer.h"
#include "Loss.h"
#include "Activation.h"
#include "NeuralNetwork.h"
#include "ModelIO.h"
using namespace std;

signed main() {
    // XOR 数据集
   
    // fstream file("ccc.out",std::ios::in|std::ios::out|std::ios::trunc);
    // file<<100<<"\n";
    //fstream file("ccc.out",std::ios::in|std::ios::out|std::ios::trunc);
    // 创建神经网络
    NeuralNetwork<double> network;
   
   
    
   // auto bias_optimizer=std::make_shared<AdamOptimizer<double,1,Eigen::Dynamic>>(Eigen::Matrix<double,1,Eigen::Dynamic>::Zero(1,4));
   network.add_module(make_shared<LinearLayer<double>>(2, 4)); // 输入层到隐藏层
    network.add_module(make_shared<ReLU<double>>());
    network.add_module(make_shared<LinearLayer<double>>(4, 1));// 隐藏层到输出层
    network.add_module(make_shared<Sigmoid<double>>());
    load_model(network,"canshu.in");
    MSELoss<double> loss;

    const int epochs = 0;
    const double learning_rate = 0.1;
    Eigen::MatrixXd input(1,2);//输出是行向量，出入还是行向量。
    Eigen::MatrixXd target(1,1);
    for (int epoch = 0; epoch < epochs; ++epoch) {
        int a=rand()%2;
        int b=rand()%2;

        
        input<<a,b;
      
        target<<(a^b);
        auto prediction = network.forward(input);
        auto loss_value = loss.forward(prediction, target);
        auto grad_output = loss.backward(prediction, target);
    
        // 添加调试输出
        // cout << "Prediction dimensions: " << prediction.rows() << "x" << prediction.cols() << endl;
        // cout << "Target dimensions: " << target.rows() << "x" << target.cols() << endl;
        // cout << "Grad output dimensions: " << grad_output.rows() << "x" << grad_output.cols() << endl;
    
        network.zero_grad();
        network.backward(grad_output);
        network.update();
    
        if (epoch % 1000 == 0) {
            cout << "Epoch " << epoch << " - Loss: " << loss_value << endl;
        }
    }
    int cnt_loss=0;
    for (int epoch = 0; epoch <10000; ++epoch) {
        int a=rand()%2;
        int b=rand()%2;

        
        input<<a,b;
        target<<(a^b);
        auto prediction = network.forward(input);
        auto loss_value = loss.forward(prediction, target);
        auto grad_output = loss.backward(prediction, target);
        if(prediction(0,0)>0.5)
        {
            if((a^b)==0)
            {
                cnt_loss++;
            }
        }
        else 
        {
            if((a^b)==1)
            cnt_loss++;
        }
        network.zero_grad();
        network.backward(grad_output);
        network.update();

        if (epoch % 1000 == 0) {
            cout << "Epoch " << epoch << " - Loss: " << cnt_loss << endl;
        }
    }

    auto final_prediction = network.forward(input);
    cout << "Final Prediction: \n" << final_prediction << endl;
    save_model(network,"canshu.in");
    return 0;
}
// template<typename Scalar>
// class NeuralNetwork : public Module<Scalar> {
// public:
//     void add_module(std::shared_ptr<Module<Scalar>> module) {
//         modules.push_back(module);
//     }

//     Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> forward(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input) override {
//         Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output = input;
//         for (auto& module : modules) {
//             output = module->forward(output);
//         }
//         return output;
//     }

//     Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> backward(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& grad_output) override {
//         Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> grad = grad_output;
//         for (auto it = modules.rbegin(); it != modules.rend(); ++it) {
//             grad = (*it)->backward(grad);
//         }
//         return grad;
//     }

//     void update() override {
//         for (auto& module : modules) {
//             module->update();
//         }
//     }

//     void zero_grad() override {
//         for (auto& module : modules) {
//             module->zero_grad();
//         }
//     }

// private:
//     std::vector<std::shared_ptr<Module<Scalar>>> modules;
// };
// //注意传入的是行向量，输出的也是行向量！
// //target也必须是行向量

// template<typename Scalar>
// class NeuralNetwork : public Module<Scalar> {
// public:
//     void add_module(std::shared_ptr<Module<Scalar>> module) {
//         modules.push_back(module);
//     }

//     Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> forward(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input) override {
//         Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output = input;
//         for (auto& module : modules) {
//             output = module->forward(output);
//         }
//         return output;
//     }

//     Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> backward(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& grad_output) override {
//         Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> grad = grad_output;
//         for (auto it = modules.rbegin(); it != modules.rend(); ++it) {
//             grad = (*it)->backward(grad);
//         }
//         return grad;
//     }

//     void update() override {
//         for (auto& module : modules) {
//             module->update();
//         }
//     }

//     void zero_grad() override {
//         for (auto& module : modules) {
//             module->zero_grad();
//         }
//     }

// private:
//     std::vector<std::shared_ptr<Module<Scalar>>> modules;
// };
// signed main() {
//     // 示例代码
//     Eigen::MatrixXd input(1, 3);
//     input << 1.0, 2.0, 3.0;

//     Eigen::MatrixXd target(1, 2);
//     target << 0.5, 1.5;

//     NeuralNetwork<double> network;
//     network.add_module(make_shared<LinearLayer<double>>(3, 2));
//     network.add_module(make_shared<ReLU<double>>());

//     MSELoss<double> loss;

//     const int epochs = 100;
//     Eigen::MatrixXd accumulated_grad_output;
//     accumulated_grad_output.setZero(1, 2); // 初始化累积梯度

//     for (int epoch = 0; epoch < epochs; ++epoch) {
//         auto prediction = network.forward(input);
//         auto loss_value = loss.forward(prediction, target);
//         auto grad_output = loss.backward(prediction, target);

//         accumulated_grad_output += grad_output; // 累积梯度

//         cout << "Epoch " << epoch + 1 << " - Loss: " << loss_value << endl;
//     }

//     accumulated_grad_output /= epochs; // 求平均梯度

//     network.backward(accumulated_grad_output); // 反向传播
//     network.update(); // 更新参数

//     auto final_prediction = network.forward(input);
//     cout << "Final Prediction: \n" << final_prediction << endl;

//     return 0;
// }

// signed main() {
//     // 示例代码
//     Eigen::MatrixXd input(1, 3);
//     input << 1.0, 2.0, 3.0;

//     Eigen::MatrixXd target(1, 2);
//     target << 0.5, 1.5;

//     LinearLayer<double> layer(3, 2);
//     MSELoss<double> loss;
//     ReLU<double> relu;

//     auto prediction = layer.forward(input);
//     auto activated_prediction = relu.forward(prediction);
//     auto loss_value = loss.forward(activated_prediction, target);
//     auto grad_output = loss.backward(activated_prediction, target);
//     auto grad_input = relu.backward(prediction, grad_output);

//     layer.backward(grad_input);
//     layer.update();

//     cout << "Prediction: \n" << prediction << endl;
//     cout << "Activated Prediction: \n" << activated_prediction << endl;
//     cout << "Loss: " << loss_value << endl;

//     return 0;
// }
// template<typename Scalar,int Rows,int Cols>
// struct Parameter{
//     Eigen::Matrix<Scalar,Rows,Cols>value;
//     Eigen::Matrix<Scalar,Rows,Cols>grad;

//     Parameter(const Eigen::Matrix<Scalar,Rows,Cols>&init_value)
//     :value(init_value)
//     {
//         grad=Eigen::Matrix<Scalar,Rows,Cols>::zero(init_value.rows(),init_value.cols());
//     }
//     Eigen::Matrix<Scalar,Rows,Cols>&get_value()
//     {
//         return value;
//     }
//     const Eigen::Matrix<Scalar,Rows,Cols>&get_value()const{
//         return value;
//     }
//     const Eigen::Matrix<Scalar,Rows,Cols>&get_grad()const
//     {
//         return grad;
//     }
//     void set_grad(const Eigen::Matrix<Scalar,Rows,Cols>&new_grad){
//         grad=new_grad;
//     }
//     void zero_grad()
//     {
//         grad.setZero();
//     }
// };
// template<typename Scalar,int Row,int Cols>
// struct Optimizer
// {
//     Parameter<Scalar,Rows,Cols>&param;
//     Optimizer (Parameter<Scalar,Rows,Cos>&P):Param(p)
//     {

//     }
//     virtual void step()=0;
// };
// template<typename Scalar,int Rows,int Cols>
// struct AdamOptimizer:Optimizer<Scalar,Rows,Cols>
// {
//     Scalar learning_rate;
//     Scalar beta1;
//     Scalar beta2;
//     Scalar epsilon;
//     int timestep;
//     Eigen::Matrix<scalar,Rows,Cols>m;
//     Eigen::Matrix<Scalar,Ros,Cols>v;

//     AdamOpimizer(Parameter<Scalar,Rows,Cols>&p,
//         Scalar lr=0.001,
//         Scalar b1=0.9,Scalar b2=0.999,Scalar eps=1e-8):
//         Optimizer<Scalar ,Rows,Cols>(p),learing_rate(lr),beta1(b1),beta2(b2),epsilon(eps),timstep(0)
//         {
//             m=Eigen::Matrix<Scalar,Rows,Cols>::Zero(p.get_grad().rows(),p.get_grad().cols());
//             v=Eigen::Matrix<Sclar,Rows,Cols>::Zero(p.get_grad().rows(),p.get_grad().cols());

//         }
//         void step()
//         {
//             timestep++;
//             auto &grad=this->param.get_grad();
//             auto &value=this->param.get_value();

//             m=beta1*m+(1-beta1)*grad;
//             v=beta2*v+(1-beta2)*grad.cwiseProduct(grad);
//             Eigen::Matrix<Scalar,Rows,Cols>m_hat=m/(1-std::pow(beta1,timestep));
//             Eigen::Matrix<Scalar,Rows,Cols>v_hat=v/(1-pow(beta2,timestep));

//             Eigen::Matrix<Scalar,Rows,Cols>epsilon_matrix(v_hat.rows(),v_hat.cols());
//             for(int i=0;i<v_hat.rows();i++)
//             {
//                 for(int j=0;j<v_hat.cols();j++)
//                 {
//                     epsilon_matrix(i,j)=epsilon;
//                 }
//             }
//             value-=learning_rate*m_hat.cwiseQuotient(v_hat.cwiseSqrt()+epsilon_matrix);
//         }
// };
// template<typename Scalar>
// struct  LinearLayer{
//     Parameter <Scalar,Eigen::Dynamic,Eigen::Dynamic>weight;
//     Parameter<Scalar,1,Eigen::Dynamic>bias;
//     AdamOptimizer<Scalar,Eigen::Dynamic,Eigen::Dynamic>weight_optimizer;
//     AdamOptimizer<Scalar,1,Eigen::Dynamic>bias_optimizer;

//     LinearLayer(int in_features,int out_features):
//     weight(Eigen::Matrix<Scalar,Eigen::Dyanmic,Eigen::Dynamic>Random(in_features,out_features)),
//     bias(Eigen::Matrix<Sclar,1,Eigen::Dynamic>::Random(1,out_features)),
//     weight_optimizer(weight),
//     bias_optimizer(bias)
//     {

//     }
//     Eigen::Matrix<Scalar,1,Eigen::Dynamic>forward(const Eigen::Matrix<Scalar,Eigne:Dynamic>&input)
//     {
//         return input*weight.get_value()+bias.get_value();
//     }
//     Eigen::Matrix<Scalar,1,Eigen::Dynamic>backward(const Eigen::Matrix<Scalar,1,Eigen::Dynamic>&input,
//     const Eigen::Matrix<Scalar,1,Eigen::Dynamic>&grad_ouput
//     )
//     {
//         Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>weight_grad=input.transpose()*grad_output;
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
//     void zero_grad(){
//         weight.zero_grad();
//         bias.zero_grad();
//     }
// };
// template<class Scalar>
// struct Base{
//     Base()
//     {
        
//     }
//     void print()
//     {
//         cout<<"Base\n";
//     }
    
// };
// template<class Scalar>
// struct Derive:Base<Scalar>
// {
//     Derive()
//     {
//        // cout<<"Derive\n";
//     }
//     void print()
//     {
//         cout<<"Derive\n";
//     }
// };
// template<typename T>
// struct c
// {
//     c()
//     {
//         T b;
//        b.print();
//     }
// };

// signed main()
// {
//     c<Base<int>> a;
// }