// BaseModel.h
#ifndef BASE_MODEL_H
#define BASE_MODEL_H

#include <Eigen/Dense>
#include <vector>
#include <stdexcept>

template<typename Scalar>
class BaseModel {
public:
    // 纯虚函数：训练模型（必须由子类实现）
    virtual void fit(
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X,
        const Eigen::VectorXi& y
    ) = 0;

    // 纯虚函数：预测类别（必须由子类实现）
    virtual Eigen::VectorXi predict(
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X
    ) = 0;

    // 评估准确率（默认实现，可被子类重写）
    virtual double score(
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X,
        const Eigen::VectorXi& y
    )  {
        if (X.rows() != y.size()) {
            throw std::invalid_argument("样本数与标签数不匹配");
        }
        Eigen::VectorXi y_pred = predict(X);
        int correct = 0;
        for (int i = 0; i < y.size(); ++i) {
            if (y_pred(i) == y(i)) correct++;
        }
        return static_cast<double>(correct) / y.size();
    }

    // 模式切换：训练模式（默认空实现，需状态管理的模型重写）
    virtual void train() {
        is_training_ = true;
    }

    // 模式切换：评估模式（默认空实现，需状态管理的模型重写）
    virtual void eval() {
        is_training_ = false;
    }

    // 获取当前模式
    bool is_training() const {
        return is_training_;
    }

    // 虚析构函数（确保子类析构正常调用）
    virtual ~BaseModel() = default;

protected:
    bool is_training_ = true;  // 默认为训练模式
};

#endif // BASE_MODEL_H