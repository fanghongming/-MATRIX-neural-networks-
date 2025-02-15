#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include "Parameter.h"
template <typename Scalar,int Rows,int Cols>
struct  Optimizer
{
    Parameter<Scalar,Rows,Cols>&param;
    Optimizer(Parameter<Scalar,Rows,Cols>&p):param(p)
    {
        
    }
    virtual void step()=0;
};
template<typename Scalar, int Rows, int Cols>
struct AdamOptimizer :Optimizer <Scalar,Rows,Cols> {

    Scalar learning_rate;
    Scalar beta1;
    Scalar beta2;
    Scalar epsilon;
    int timestep;
    Eigen::Matrix<Scalar, Rows, Cols> m;
    Eigen::Matrix<Scalar, Rows, Cols> v;
    AdamOptimizer(Parameter<Scalar, Rows, Cols>& p,
         Scalar lr = 0.001,
                  Scalar b1 = 0.9, Scalar b2 = 0.999, Scalar eps = 1e-8):
                  Optimizer<Scalar,Rows,Cols>(p), learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), timestep(0) 
        {
       
         m = Eigen::Matrix<Scalar, Rows, Cols>::Zero(p.get_grad().rows(),p.get_grad().cols());
         v = Eigen::Matrix<Scalar, Rows, Cols>::Zero(p.get_grad().rows(),p.get_grad().cols());
        }
    void step() {
        timestep++;
        auto& grad = this->param.get_grad();
        auto& value = this->param.get_value();

        // 计算一阶矩估计
        m = beta1 * m + (1 - beta1) * grad;
        // 计算二阶矩估计
        v = beta2 * v + (1 - beta2) * grad.cwiseProduct(grad);

        // 修正一阶矩估计的偏差
        Eigen::Matrix<Scalar, Rows, Cols> m_hat = 
        m / (1 - std::pow(beta1, timestep));
        // 修正二阶矩估计的偏差
        Eigen::Matrix<Scalar, Rows, Cols> v_hat = v / (1 - std::pow(beta2, timestep));

        // 将epsilon转换为矩阵
        //Eigen::Matrix<Scalar, Rows, Cols> epsilon_matrix = Eigen::Matrix<Scalar, Rows, Cols>::Constant(epsilon);
        Eigen::Matrix<Scalar,Rows,Cols>epsilon_matrix(v_hat.rows(),v_hat.cols());
        for(int i=0;i<v_hat.rows();i++)
        {
            for(int j=0;j<v_hat.cols();j++)
            {
                epsilon_matrix(i,j)=epsilon;
            }
        }
        value -= learning_rate * m_hat.cwiseQuotient(v_hat.cwiseSqrt() + epsilon_matrix);
    }
};
#endif 
