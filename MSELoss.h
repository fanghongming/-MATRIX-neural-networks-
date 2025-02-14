#ifndef MSELOSS_H
#define MSELOSS_H
#include"Loss.h"
template<typename Scalar>
struct MSELoss : public Loss<Scalar> {
    Scalar forward(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& prediction,
                   const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& target) override {
        return (prediction - target).array().square().mean();
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> backward(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& prediction,
                                                                   const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& target) override {
        return 2 * (prediction - target) / prediction.size();//求导
    }
};
#endif 