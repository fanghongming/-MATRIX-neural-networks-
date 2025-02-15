#ifndef PARAMETER_H
#define PARAMETER_H
#include<Eigen/Dense>
template<typename Scalar, int Rows, int Cols>
class Parameter {
private:
    Eigen::Matrix<Scalar, Rows, Cols> value;  // 存储参数值
    Eigen::Matrix<Scalar, Rows, Cols> grad;   // 存储梯度值

public:
    // 构造函数，初始化参数值并将梯度初始化为零矩阵
    Parameter(const Eigen::Matrix<Scalar, Rows, Cols>& init_value)//Parmater此时必须是一个左值
        : value(init_value)
         {
            
                //std::cout<<Rows<<" "<<Cols<<"\n";
                // 固定大小矩阵，使用静态方法初始化
                grad = Eigen::Matrix<Scalar, Rows, Cols>::Zero(init_value.rows(),init_value.cols());
            // 是用于创建一个全零矩阵的静态方法。当 Rows 或者 Cols 
            // 被设定为 Eigen::Dynamic 时，调用这个方法就会触发静态断言错误。
            // 这是因为 Zero() 这个方法在某些实现中可能是针对固定大小矩阵设计的，不支持动态大小的矩阵。
            // 解决方案
        }

    // 获取参数值，返回非const引用
    Eigen::Matrix<Scalar, Rows, Cols>& get_value() {
        return value;
    }

    // 获取参数值的const版本，用于只读访问
    const Eigen::Matrix<Scalar, Rows, Cols>& get_value() const {
        return value;
    }

    // 获取梯度值
    const Eigen::Matrix<Scalar, Rows, Cols>& get_grad() const {
        return grad;
    }

    // 设置梯度值
    void set_grad(const Eigen::Matrix<Scalar, Rows, Cols>& new_grad) {
        grad = new_grad;
    }

    // 清零梯度
    void zero_grad() {
        grad.setZero();
    }
};


#endif 