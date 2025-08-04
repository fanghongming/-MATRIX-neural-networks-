#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <random>
#include <map>
#include "BaseModel.h"
// 节点结构定义
struct Node {
    int feature_idx;          // 用于分裂的特征索引
    double threshold;         // 分裂阈值
    int class_label;          // 叶子节点的类别标签
    double impurity;          // 节点不纯度
    int n_samples;            // 节点包含的样本数
    Node* left;               // 左子树
    Node* right;              // 右子树
    
    // 构造函数
    Node() : feature_idx(-1), threshold(0.0), class_label(-1),
             impurity(0.0), n_samples(0), left(nullptr), right(nullptr) {}
};

template <typename Scalar = double>
class DecisionTree:public BaseModel<Scalar> {
private:
    Node* root;               // 树根节点
    int max_depth;            // 树的最大深度
    int min_samples_split;    // 最小分裂样本数
    int min_samples_leaf;     // 叶子节点最小样本数
    int max_features;         // 分裂时考虑的最大特征数
    bool random_state;        // 随机状态标志

    // 计算Gini不纯度
    double gini_impurity(const Eigen::VectorXi& y) const {
        const int n_samples = y.size();
        if (n_samples == 0) return 0.0;

        std::map<int, int> counts;
        for (int i = 0; i < n_samples; ++i) {
            counts[y(i)]++;
        }

        double impurity = 1.0;
        for (const auto& pair : counts) {
            double p = static_cast<double>(pair.second) / n_samples;
            impurity -= p * p;
        }
        return impurity;
    }

    // 找到出现次数最多的类别
    int most_frequent_class(const Eigen::VectorXi& y) const {
        std::map<int, int> counts;
        for (int i = 0; i < y.size(); ++i) {
            counts[y(i)]++;
        }

        int max_count = -1;
        int best_class = -1;
        for (const auto& pair : counts) {
            if (pair.second > max_count) {
                max_count = pair.second;
                best_class = pair.first;
            }
        }
        return best_class;
    }

    // 分割数据集
    void split(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X, 
               const Eigen::VectorXi& y, int feature_idx, double threshold,
               Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X_left,
               Eigen::VectorXi& y_left,
               Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X_right,
               Eigen::VectorXi& y_right) const {
        
        std::vector<int> left_indices, right_indices;
        for (int i = 0; i < X.rows(); ++i) {
            if (X(i, feature_idx) <= threshold) {
                left_indices.push_back(i);
            } else {
                right_indices.push_back(i);
            }
        }

        // 构建左子树数据集
        X_left.resize(left_indices.size(), X.cols());
        y_left.resize(left_indices.size());
        for (int i = 0; i < left_indices.size(); ++i) {
            X_left.row(i) = X.row(left_indices[i]);
            y_left(i) = y(left_indices[i]);
        }

        // 构建右子树数据集
        X_right.resize(right_indices.size(), X.cols());
        y_right.resize(right_indices.size());
        for (int i = 0; i < right_indices.size(); ++i) {
            X_right.row(i) = X.row(right_indices[i]);
            y_right(i) = y(right_indices[i]);
        }
    }

    // 计算分裂后的不纯度增益
    double information_gain(const Eigen::VectorXi& y, 
                           const Eigen::VectorXi& y_left, 
                           const Eigen::VectorXi& y_right) const {
        const double parent_impurity = gini_impurity(y);
        const int n = y.size();
        const int n_left = y_left.size();
        const int n_right = y_right.size();

        if (n_left == 0 || n_right == 0) return 0.0;

        const double child_impurity = 
            (static_cast<double>(n_left) / n) * gini_impurity(y_left) +
            (static_cast<double>(n_right) / n) * gini_impurity(y_right);

        return parent_impurity - child_impurity;
    }

    // 找到最佳分裂点
    void find_best_split(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X, 
                        const Eigen::VectorXi& y, int& best_feature, 
                        double& best_threshold, double& best_gain) const {
        
        best_gain = -1.0;
        best_feature = -1;
        best_threshold = 0.0;

        const int n_samples = X.rows();
        const int n_features = X.cols();

        // 随机选择特征子集
        std::vector<int> features(n_features);
        std::iota(features.begin(), features.end(), 0);
        if (max_features < n_features && max_features > 0) {
            std::shuffle(features.begin(), features.end(), std::mt19937(std::random_device{}()));
            features.resize(max_features);
        }

        for (int feature : features) {
            // 获取该特征的所有值并去重排序
            Eigen::Vector<Scalar, Eigen::Dynamic> values = X.col(feature);
            std::sort(values.data(), values.data() + values.size());
            
            // 尝试可能的分裂阈值
            for (int i = 1; i < n_samples; ++i) {
                if (values(i) == values(i-1)) continue;
                
                double threshold = (values(i) + values(i-1)) / 2.0;
                
                // 分裂数据集
                Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> X_left, X_right;
                Eigen::VectorXi y_left, y_right;
                split(X, y, feature, threshold, X_left, y_left, X_right, y_right);
                
                // 计算信息增益
                double gain = information_gain(y, y_left, y_right);
                
                // 更新最佳分裂
                if (gain > best_gain) {
                    best_gain = gain;
                    best_feature = feature;
                    best_threshold = threshold;
                }
            }
        }
    }

    // 递归构建树
    Node* build_tree(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X, 
                    const Eigen::VectorXi& y, int depth) {
        
        Node* node = new Node();
        node->n_samples = X.rows();
        node->impurity = gini_impurity(y);
        node->class_label = most_frequent_class(y);

        // 停止条件
        if (depth >= max_depth || 
            X.rows() < min_samples_split || 
            node->impurity == 0.0) {
            return node;
        }

        // 寻找最佳分裂
        int best_feature;
        double best_threshold;
        double best_gain;
        find_best_split(X, y, best_feature, best_threshold, best_gain);

        // 如果没有找到有增益的分裂，停止分裂
        if (best_feature == -1 || best_gain <= 0) {
            return node;
        }

        // 分裂数据集
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> X_left, X_right;
        Eigen::VectorXi y_left, y_right;
        split(X, y, best_feature, best_threshold, X_left, y_left, X_right, y_right);

        // 检查子节点样本数是否满足最小要求
        if (X_left.rows() < min_samples_leaf || X_right.rows() < min_samples_leaf) {
            return node;
        }

        // 递归构建左右子树
        node->feature_idx = best_feature;
        node->threshold = best_threshold;
        node->left = build_tree(X_left, y_left, depth + 1);
        node->right = build_tree(X_right, y_right, depth + 1);

        return node;
    }

    // 递归预测单个样本
    int predict_sample(const Node* node, const Eigen::Matrix<Scalar, 1, Eigen::Dynamic>& x) const {
        // 如果是叶子节点，返回类别
        if (node->left == nullptr && node->right == nullptr) {
            return node->class_label;
        }

        // 否则递归预测
        if (x(node->feature_idx) <= node->threshold) {
            return predict_sample(node->left, x);
        } else {
            return predict_sample(node->right, x);
        }
    }

    // 释放树内存
    void destroy_tree(Node* node) {
        if (node == nullptr) return;
        destroy_tree(node->left);
        destroy_tree(node->right);
        delete node;
    }

public:
    // 构造函数
    DecisionTree(int max_depth = 5, 
                          int min_samples_split = 2,
                          int min_samples_leaf = 1,
                          int max_features = 0,
                          bool random_state = false)
        : max_depth(max_depth), 
          min_samples_split(min_samples_split),
          min_samples_leaf(min_samples_leaf),
          max_features(max_features),
          random_state(random_state),
          root(nullptr) {
        
        if (max_depth <= 0) throw std::invalid_argument("max_depth must be positive");
        if (min_samples_split <= 0) throw std::invalid_argument("min_samples_split must be positive");
        if (min_samples_leaf <= 0) throw std::invalid_argument("min_samples_leaf must be positive");
    }

    // 析构函数
    ~DecisionTree() {
        destroy_tree(root);
    }

    // 训练模型
    void fit(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X, 
            const Eigen::VectorXi& y)override 
            {
        if (X.rows() != y.size()) {
            throw std::invalid_argument("X and y must have the same number of samples");
        }
        if (X.rows() == 0) {
            throw std::invalid_argument("No samples provided");
        }

        // 如果max_features为0，则使用所有特征
        if (max_features <= 0) {
            max_features = X.cols();
        }

        // 销毁已有树
        destroy_tree(root);

        // 构建新树
        root = build_tree(X, y, 0);
    }

   // 批量预测（批量样本）
    Eigen::VectorXi predict(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X)  override
     {
        Eigen::VectorXi predictions(X.rows());
        for (int i = 0; i < X.rows(); ++i) {
            // 对每个样本调用单个预测方法
            predictions(i) = predict_single(X.row(i));
        }
        return predictions;
        }
        int predict_single(const Eigen::Matrix<Scalar, 1, Eigen::Dynamic>& x) const {
        if (root == nullptr) {
            throw std::runtime_error("Model not fitted. Call fit first.");
        }
        return predict_sample(root, x);  // 复用原递归逻辑
    }
};

#endif // DECISION_TREE_H
    