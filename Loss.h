#ifndef LOSS_H
#define LOSS_H
#include<Eigen/Dense>
#include "Module.h"
template<typename Scalar>
struct Loss :public Module<Scalar>{
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> 
    forward(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input) override{
        return input;
    }
     Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> backward(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& grad_output) override
     {
        return grad_output;
     }
    virtual Scalar forward(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& prediction,
                           const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& target) = 0;

    virtual Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> backward(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& prediction,
                                                                           const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& target) = 0;
    void update()override
    {

    }
    void zero_grad()override
    {

    }
              
};
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
    void update()override
    {

    }
    void zero_grad()override
    {

    }
          
};
#endif 