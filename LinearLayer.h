#ifndef LINEARLAYER
#define  LINEARLAYER
#include "Parameter.h"
#include "AdamOptimizer.h"
template<typename Scalar>
class LinearLayer {
private:
    Parameter<Scalar, Eigen::Dynamic, Eigen::Dynamic> weight;
    Parameter<Scalar, 1, Eigen::Dynamic> bias;
    AdamOptimizer<Scalar, Eigen::Dynamic, Eigen::Dynamic> weight_optimizer;
    AdamOptimizer<Scalar, 1, Eigen::Dynamic> bias_optimizer;

public:
    LinearLayer(int in_features, int out_features)
        : weight(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Random(in_features, out_features)),
          bias(Eigen::Matrix<Scalar, 1, Eigen::Dynamic>::Random(1, out_features)),
          weight_optimizer(weight),
          bias_optimizer(bias) {
          }

    // 前向传播
    Eigen::Matrix<Scalar, 1, Eigen::Dynamic> forward(const Eigen::Matrix<Scalar, 1, Eigen::Dynamic>& input) {
        return input * weight.get_value() + bias.get_value();
    }

    // 反向传播
    Eigen::Matrix<Scalar, 1, Eigen::Dynamic> backward(const Eigen::Matrix<Scalar, 1, Eigen::Dynamic>& input,
                                                      const Eigen::Matrix<Scalar, 1, Eigen::Dynamic>& grad_output) {
        // 计算权重的梯度
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> weight_grad = input.transpose() * grad_output;
        weight.set_grad(weight_grad);

        // 计算偏置的梯度
        Eigen::Matrix<Scalar, 1, Eigen::Dynamic> bias_grad = grad_output;
        bias.set_grad(bias_grad);

        // 计算输入的梯度
        Eigen::Matrix<Scalar, 1, Eigen::Dynamic> grad_input = grad_output * weight.get_value().transpose();

        return grad_input;
    }

    // 更新参数
    void update() {
         weight_optimizer.step();
         bias_optimizer.step();
    }

    // 清零梯度
    void zero_grad() {
        weight.zero_grad();
        bias.zero_grad();
    }
};
#endif 