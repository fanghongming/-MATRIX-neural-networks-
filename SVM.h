#ifndef SVM_H
#define SVM_H

#include <Eigen/Dense>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <random>
#include "BaseModel.h"

template <typename Scalar = double>
class SVC :public BaseModel<Scalar>
{
private:
    int max_iter_;          // 最大迭代次数
    double C_;              // 惩罚参数
    double tol_;            // 收敛容忍度
    Eigen::Vector<Scalar, Eigen::Dynamic> alpha_;  // 拉格朗日乘子
    Eigen::Vector<Scalar, Eigen::Dynamic> w_;      // 权重向量
    Scalar b_;              // 偏置项
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> X_;  // 支持向量
    Eigen::VectorXi y_;     // 支持向量标签

    // 线性核函数
    Scalar kernel(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x1, 
                 const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x2) const {
        return x1.dot(x2);  // 线性核: K(x1,x2) = x1·x2
    }

    // 计算决策函数值 f(x) = w·x + b
    Scalar decision_function(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x) const {
        return w_.dot(x) + b_;
    }

    // 随机选择第二个alpha的索引
    int select_second_alpha(int i, int n_samples) const {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(0, n_samples - 1);
        
        int j = i;
        while (j == i) {
            j = dist(gen);
        }
        return j;
    }

public:
    /**
     * @brief 构造函数
     * @param C 惩罚参数，控制误分类的惩罚程度，C越大惩罚越重
     * @param max_iter 最大迭代次数
     * @param tol 收敛容忍度
     */
    SVC(double C = 1.0, int max_iter = 1000, double tol = 1e-3)
        : C_(C), max_iter_(max_iter), tol_(tol), b_(0.0) {
        if (C <= 0) {
            throw std::invalid_argument("惩罚参数C必须为正数");
        }
        if (max_iter <= 0) {
            throw std::invalid_argument("最大迭代次数必须为正数");
        }
        if (tol <= 0) {
            throw std::invalid_argument("容忍度必须为正数");
        }
    }

    /**
     * @brief 训练SVM模型（使用SMO算法）
     * @param X 输入特征矩阵，每行一个样本
     * @param y 标签向量，取值应为+1或-1
     */
    void fit(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X, 
             const Eigen::VectorXi& y)override {
        const int n_samples = X.rows();
        const int n_features = X.cols();

        // 检查输入有效性
        if (n_samples != y.size()) {
            throw std::invalid_argument("样本数与标签数不匹配");
        }
        for (int i = 0; i < n_samples; ++i) {
            if (y(i) != 1 && y(i) != -1) {
                throw std::invalid_argument("标签必须为+1或-1");
            }
        }

        // 初始化参数
        alpha_ = Eigen::Vector<Scalar, Eigen::Dynamic>::Zero(n_samples);
        w_ = Eigen::Vector<Scalar, Eigen::Dynamic>::Zero(n_features);
        b_ = 0.0;

        int iter = 0;
        bool changed = false;

        // 简化版SMO算法
        while (iter < max_iter_ && (iter == 0 || changed)) {
            changed = false;
            for (int i = 0; i < n_samples; ++i) {
                // 计算当前样本的预测值
                Scalar f_i = decision_function(X.row(i));
                Scalar E_i = f_i - y(i);  // 误差

                // 检查是否违反KKT条件
                if ((y(i) * E_i < -tol_ && alpha_(i) < C_) || 
                    (y(i) * E_i > tol_ && alpha_(i) > 0)) {
                    
                    // 选择第二个样本
                    int j = select_second_alpha(i, n_samples);
                    Scalar f_j = decision_function(X.row(j));
                    Scalar E_j = f_j - y(j);

                    // 保存旧的alpha值
                    Scalar alpha_i_old = alpha_(i);
                    Scalar alpha_j_old = alpha_(j);

                    // 计算上下界
                    Scalar L, H;
                    if (y(i) != y(j)) {
                        L = std::max(0.0, alpha_j_old - alpha_i_old);
                        H = std::min(C_, C_ + alpha_j_old - alpha_i_old);
                    } else {
                        L = std::max(0.0, alpha_i_old + alpha_j_old - C_);
                        H = std::min(C_, alpha_i_old + alpha_j_old);
                    }

                    if (std::abs(L - H) < 1e-9) {
                        continue;
                    }

                    // 计算核函数值
                    Scalar K_ii = kernel(X.row(i), X.row(i));
                    Scalar K_jj = kernel(X.row(j), X.row(j));
                    Scalar K_ij = kernel(X.row(i), X.row(j));
                    Scalar eta = K_ii + K_jj - 2 * K_ij;

                    if (eta <= 0) {
                        continue;  // 不符合要求，跳过
                    }

                    // 更新alpha_j
                    alpha_(j) += y(j) * (E_i - E_j) / eta;
                    
                    // 裁剪alpha_j到[L, H]范围内
                    if (alpha_(j) > H) alpha_(j) = H;
                    else if (alpha_(j) < L) alpha_(j) = L;

                    // 检查alpha_j是否有显著变化
                    if (std::abs(alpha_(j) - alpha_j_old) < 1e-9) {
                        continue;
                    }

                    // 更新alpha_i
                    alpha_(i) += y(i) * y(j) * (alpha_j_old - alpha_(j));

                    // 更新偏置项b
                    Scalar b1 = b_ - E_i - y(i) * (alpha_(i) - alpha_i_old) * K_ii 
                              - y(j) * (alpha_(j) - alpha_j_old) * K_ij;
                    Scalar b2 = b_ - E_j - y(i) * (alpha_(i) - alpha_i_old) * K_ij 
                              - y(j) * (alpha_(j) - alpha_j_old) * K_jj;

                    if (alpha_(i) > 0 && alpha_(i) < C_) {
                        b_ = b1;
                    } else if (alpha_(j) > 0 && alpha_(j) < C_) {
                        b_ = b2;
                    } else {
                        b_ = (b1 + b2) / 2;
                    }

                    changed = true;
                }
            }
            iter++;
        }

        // 计算权重向量w（仅适用于线性核）
        for (int i = 0; i < n_samples; ++i) {
            if (alpha_(i) > 1e-9) {  // 支持向量
                w_ += alpha_(i) * y(i) * X.row(i).transpose();
            }
        }

        // 保存支持向量
        std::vector<int> support_indices;
        for (int i = 0; i < n_samples; ++i) {
            if (alpha_(i) > 1e-9) {
                support_indices.push_back(i);
            }
        }

        X_ = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>(support_indices.size(), n_features);
        y_ = Eigen::VectorXi(support_indices.size());
        
        for (int i = 0; i < support_indices.size(); ++i) {
            X_.row(i) = X.row(support_indices[i]);
            y_(i) = y(support_indices[i]);
        }
    }

    /**
     * @brief 预测样本类别
     * @param X 输入特征矩阵，每行一个样本
     * @return 预测的类别标签（+1或-1）
     */
    Eigen::VectorXi predict(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X)  override {
        if (w_.size() == 0) {
            throw std::runtime_error("模型未训练，请先调用fit方法");
        }

        const int n_samples = X.rows();
        Eigen::VectorXi y_pred(n_samples);

        for (int i = 0; i < n_samples; ++i) {
            Scalar score = decision_function(X.row(i));
            y_pred(i) = (score >= 0) ? 1 : -1;
        }

        return y_pred;
    }

    /**
     * @brief 计算模型在测试集上的准确率
     * @param X 测试特征矩阵
     * @param y 测试标签向量
     * @return 准确率（正确预测的样本比例）
     */
    double score(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X, 
                 const Eigen::VectorXi& y) const {
        if (X.rows() != y.size()) {
            throw std::invalid_argument("样本数与标签数不匹配");
        }

        Eigen::VectorXi y_pred = predict(X);
        int correct = 0;
        for (int i = 0; i < y.size(); ++i) {
            if (y_pred(i) == y(i)) {
                correct++;
            }
        }

        return static_cast<double>(correct) / y.size();
    }

    /**
     * @brief 获取支持向量
     * @return 支持向量矩阵，每行一个支持向量
     */
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& support_vectors() const {
        return X_;
    }

    /**
     * @brief 获取权重向量（仅适用于线性核）
     * @return 权重向量
     */
    const Eigen::Vector<Scalar, Eigen::Dynamic>& coef_() const {
        return w_;
    }

    /**
     * @brief 获取偏置项
     * @return 偏置值b
     */
    Scalar intercept_() const {
        return b_;
    }
};

#endif // SVM_H
    