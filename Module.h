#ifndef MODULE_H
#define MODULE_H

#include <Eigen/Dense>
#include <vector>
#include <memory>

template<typename Scalar>
class Module {
public:
    virtual Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> forward(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input) = 0;
    virtual Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> backward(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& grad_output) = 0;
    virtual void update() = 0;
    virtual void zero_grad() = 0;
};

#endif // MODULE_H