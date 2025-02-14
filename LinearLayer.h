#ifndef LINEARLAYER_H
#define LINEARLAYER_H

#include "Module.h"
#include "Parameter.h"
#include "AdamOptimizer.h"

template<typename Scalar>
class LinearLayer : public Module<Scalar> {
public:
    Parameter<Scalar, Eigen::Dynamic, Eigen::Dynamic> weight;
    Parameter<Scalar, 1, Eigen::Dynamic> bias;
    AdamOptimizer<Scalar, Eigen::Dynamic, Eigen::Dynamic> weight_optimizer;
    AdamOptimizer<Scalar, 1, Eigen::Dynamic> bias_optimizer;

    LinearLayer(int in_features, int out_features)
        : weight(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Random(in_features, out_features)),
          bias(Eigen::Matrix<Scalar, 1, Eigen::Dynamic>::Random(1, out_features)),
          weight_optimizer(weight),
          bias_optimizer(bias) {}

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> forward(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input) override {
        this->input = input;
        return input * weight.get_value() + bias.get_value();
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> backward(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& grad_output) override {
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

private:
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> input;
};

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
