#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <Eigen/Dense>
#include "Module.h"

template<typename Scalar>
class ReLU : public Module<Scalar> {
public:
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> forward(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input) override {
        this->input = input;
        return input.cwiseMax(0);
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> backward(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& grad_output) override {
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> grad_input = input;
        grad_input = (input.array() > 0).select(grad_output, 0);
        return grad_input;
    }

    void update() override {
        // ReLU 没有参数需要更新
    }

    void zero_grad() override {
        // ReLU 没有梯度需要清零
    }

private:
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> input;
};

template<typename Scalar>
class Sigmoid : public Module<Scalar> {
public:
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> forward(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input) override {
        this->input = input;
        return 1 / (1 + (-input.array()).exp());
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> backward(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& grad_output) override {
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> sigmoid = forward(input);

        return grad_output.cwiseProduct(sigmoid.cwiseProduct(Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>::Ones(sigmoid.rows(),sigmoid.cols()) - sigmoid));
    }

    void update() override {
        // Sigmoid 没有参数需要更新
    }

    void zero_grad() override {
        // Sigmoid 没有梯度需要清零
    }

private:
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> input;
};

#endif // ACTIVATION_H