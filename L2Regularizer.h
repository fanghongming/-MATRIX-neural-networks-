#ifndef L2REGULARIZER_H
#define L2REGULARIZER_H

#include <Eigen/Dense>

template<typename Scalar>
class L2Regularizer {
public:
    // lambda 为正则化系数
    L2Regularizer(Scalar lambda) : lambda(lambda) {}

    // 计算正则化损失：lambda * ||param||²
    Scalar loss(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& param) const {
        return lambda * param.squaredNorm();
    }

    // 计算正则化梯度：2 * lambda * param
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> grad(
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& param) const {
        return 2 * lambda * param;
    }

private:
    Scalar lambda;
};

#endif // L2REGULARIZER_H