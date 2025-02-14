#ifndef LOSS_H
#define LOSS_H
#include<Eigen/Dense>
template<typename Scalar>
struct Loss {
    virtual Scalar forward(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& prediction,
                           const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& target) = 0;

    virtual Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> backward(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& prediction,
                                                                           const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& target) = 0;
};
#endif 