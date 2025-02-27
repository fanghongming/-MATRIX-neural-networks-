#ifndef DROPOUT_H
#define DROPOUT_H

#include "Module.h"
#include <Eigen/Dense>
#include <random>

template<typename Scalar>
class Dropout : public Module<Scalar> {
public:
    Dropout(Scalar keep_prob)
        : keep_prob(keep_prob),
          training(true),
          engine(std::random_device{}()),
          dist(0.0, 1.0) { }

    void set_training(bool mode) override {
        training = mode;
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> forward(
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input) override {
        this->input = input;
        if (training) {
            dropout_mask = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::NullaryExpr(
                input.rows(), input.cols(), [&]() { return (dist(engine) < keep_prob) ? Scalar(1) : Scalar(0); }
            );
            return (input.array() * dropout_mask.array()) / keep_prob;
        } else {
            return input;
        }
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> backward(
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& grad_output) override {
        if (training) {
            return (grad_output.array() * dropout_mask.array()) / keep_prob;
        } else {
            return grad_output;
        }
    }

    void update() override { }

    void zero_grad() override { }

private:
    Scalar keep_prob;
    bool training;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> dropout_mask;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> input;
    std::mt19937 engine;
    std::uniform_real_distribution<Scalar> dist;
};

#endif