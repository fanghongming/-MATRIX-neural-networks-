#ifndef LINEARLAYER_H
#define LINEARLAYER_H

#include "Module.h"
#include "Parameter.h"
#include "AdamOptimizer.h"
#include <fstream>
//如果完全在内部操作，使用模板，否则，有一部分在外部操作，使用指针，或者引用
template<typename Scalar,typename Optimizer_w=AdamOptimizer<double,Eigen::Dynamic,Eigen::Dynamic>,typename Optimizer_b=AdamOptimizer<double,1,Eigen::Dynamic>>
class LinearLayer : public Module<Scalar> {
public:
    Parameter<Scalar, Eigen::Dynamic, Eigen::Dynamic> weight;
    Parameter<Scalar, 1, Eigen::Dynamic> bias;
    Optimizer_w  weight_optimizer;
    Optimizer_b  bias_optimizer;
    LinearLayer(int in_features, int out_features)//传入两四个数，第一个数是输入层的个数，第二个数是输出层的个数，第三个数是w的优化器，第四个数是b的优化器。
        : weight(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Random(in_features, out_features)),
          bias(Eigen::Matrix<Scalar, 1, Eigen::Dynamic>::Random(1, out_features)),
          weight_optimizer(weight),
          bias_optimizer(bias) {
          }

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> forward
    (const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input) override {
        this->input = input;
        return input * weight.get_value() + bias.get_value();
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> backward
    (const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& grad_output) override {
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> weight_grad = input.transpose() * grad_output;
        weight.set_grad(weight_grad);

        Eigen::Matrix<Scalar, 1, Eigen::Dynamic> bias_grad = grad_output;
        bias.set_grad(bias_grad);

        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> grad_input = grad_output * weight.get_value().transpose();
        return grad_input;
    }

    void update() override {
        weight_optimizer.step();
        bias_optimizer.step();
    }

    void zero_grad() override {
        weight.zero_grad();
        bias.zero_grad();
    }
    void save(std::fstream& file)override
    {
        int rows=weight.get_value().rows();
        int cols=weight.get_value().cols();
        file<<rows<<" "<<cols<<" ";
        file<<"\n";
        for(int i=0;i<rows;i++)
        {
            for(int j=0;j<cols;j++)
            {
               file<<weight.get_value()(i,j)<<" ";
               //std::cerr<<weight.get_value()(i,j)<<" ";
            }
            file<<"\n";
            //std::cerr<<"\n";
        }
        rows=bias.get_value().rows();
        cols=bias.get_value().cols();
        file<<rows<<" "<<cols<<" ";
        file<<"\n";
        for(int i=0;i<rows;i++)
        {
            for(int j=0;j<cols;j++)
            {
                file<<bias.get_value()(i,j)<<" ";
            }
            file<<"\n";
        }
    }
    void load(std::fstream&file)override
    {
        int rows;
        int cols;
        file>>rows>>cols;
        for(int i=0;i<rows;i++)
        {
            for(int j=0;j<cols;j++)
            {
                Scalar t;
                file>>t;
                weight.get_value()(i,j)=t;
                //std::cerr<<t<<" "<<weight.get_value()<<"\n";
            }
        }
        file>>rows>>cols;
        for(int i=0;i<rows;i++)
        {
            for(int j=0;j<cols;j++)
            {
                Scalar t;
                file>>t;
                bias.get_value()(i,j)=t;
            }
        }
    }
private:
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> input;
};
// template<typename Scalar>
// class LinearLayer : public Module<Scalar> {
// public:
//     Parameter<Scalar, Eigen::Dynamic, Eigen::Dynamic> weight;
//     Parameter<Scalar, 1, Eigen::Dynamic> bias;
//     AdamOptimizer<Scalar, Eigen::Dynamic, Eigen::Dynamic> weight_optimizer;
//     AdamOptimizer<Scalar, 1, Eigen::Dynamic> bias_optimizer;

//     LinearLayer(int in_features, int out_features)
//         : weight(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Random(in_features, out_features)),
//           bias(Eigen::Matrix<Scalar, 1, Eigen::Dynamic>::Random(1, out_features)),
//           weight_optimizer(weight),
//           bias_optimizer(bias) {}

//     Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> forward
//     (const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input) override {
//         this->input = input;
//         return input * weight.get_value() + bias.get_value();
//     }

//     Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> backward
//     (const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& grad_output) override {
//         Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> weight_grad = input.transpose() * grad_output;
//         weight.set_grad(weight_grad);

//         Eigen::Matrix<Scalar, 1, Eigen::Dynamic> bias_grad = grad_output;
//         bias.set_grad(bias_grad);

//         Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> grad_input = grad_output * weight.get_value().transpose();
//         return grad_input;
//     }

//     void update() override {
//         weight_optimizer.step();
//         bias_optimizer.step();
//     }

//     void zero_grad() override {
//         weight.zero_grad();
//         bias.zero_grad();
//     }

// private:
//     Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> input;
// };

#endif // LINEARLAYER_H
 //w[i][j]表示从前一层的 i节点到后一层j节点的权重 
    //w[1][1] w[1][2] w[1][3] w[1][4]
    //w[2][1] w[2][2] w[2][3] w[2][4]
    //w[3][1] w[3][2] w[3][3] w[3][4]
    //转置为
    //w[1][1] w[2][1] w[3][1]
    //w[1][2] w[2][2] w[3][2]
    //w[1][3] w[2][3] w[3][3]
    //w[1][4] w[2][4] w[3][4]
    //grad_output=a[1] a[2] a[3] a[4]
    //grad_output[1][1]=w[1][1]*a[1]+w[1][2]*(d a[2]/d(sigma(w[1][1]*pre[1]+w[2][1]*pre[1])))a[2]*+w[1][3]*a[3]+w[1][4]*a[4]
// template<typename Scalar>
// class LinearLayer:public Modlue<Scalar>
// {
//     public:
//     Parameter<Scalar,Eigen::Dynamic,Eigen::Dynamic>weight;
//     Parameter<Scalar,1,Eigen::Dynamic>bias;

//     std::shared_ptr<Optimizer<Scalar,Eigen::Dynamic,Eigen::Dynamic>>weight_optimizer;
//     std::shared_ptr<Optimizer<Scalar,1,Eigen::Dynamic>>bias_optimizer;

//     LinearLayer(int in_features,int out_features
//     ,  std::shared_ptr<Optimizer<Scalar,Eigen::Dynamci,Eigen::Dynamci>>weight_opt,
//     std::shared_ptr<Optimizer<Scalar,1,Eigen::Dynamci>>bias_opt)
//     :weight_optimizer(weight_opt),
//     bias_optimizer(bias_opt)
//     {

//     }
//     Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamci>forward
//     (const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>&input)override{
//         this->input=input;
//         return input*weight.get_value()+bias.get_value();
//     }
//     Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>backward
//     (const Eigen::Matrix<Scalar,EIgen::Dynamci,Eigen::Dynamic>&grad_output)override{
//         Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>weight_grad=input.transpose()*grad_output;
//         weight.set_gra(weight_grad);

//         Eigen::Matrix<Scalar,1,Eigen::Dynamic>bias_grad=grad_output;
//         bias.set_grad(bias_grad);

//         Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamci>grad_input=grad_output*weight.get_value().transpose();
//         return grad_input;
//     }
//     void upate()override{
//         weight_optimizer->step();
//         bias_optimizer->step();
//     }
//     void zero_grad()override{
//         weight.zero_grad();
//         bias.zero_grad();
//     }
//     private:
//     Eigen::Matrix<Scalar,Eigen::Dynamc,Eigen::Dynamic>input;
// };