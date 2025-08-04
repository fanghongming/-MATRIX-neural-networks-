#ifndef KMEANS_H
#define KMEANS_H

#include <Eigen/Dense>
#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>


template <typename Scalar = double>
class KMeans {
private:
    int n_clusters_;          // 聚类数量
    int max_iter_;            // 最大迭代次数
    double tol_;              // 收敛阈值
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> centroids_;  // 聚类中心
    Eigen::VectorXi labels_;  // 最后一次拟合的标签
    double inertia_;          // 惯性值（所有样本到其聚类中心的距离平方和）

    // 计算欧氏距离的平方
    Scalar distance_squared(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& a, 
                           const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& b) const {
        return (a - b).squaredNorm();
    }

    // 初始化聚类中心（k-means++算法）
    void initialize_centroids_kmeans_plus_plus(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X) {
        const int n_samples = X.rows();
        const int n_features = X.cols();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(0, n_samples - 1);

        // 随机选择第一个中心
        centroids_.row(0) = X.row(dist(gen));

        // 选择剩余中心
        for (int c = 1; c < n_clusters_; ++c) {
            // 计算每个样本到最近中心的距离平方
            Eigen::Vector<Scalar, Eigen::Dynamic> distances(n_samples);
            for (int i = 0; i < n_samples; ++i) {
                Scalar min_dist = std::numeric_limits<Scalar>::max();
                for (int j = 0; j < c; ++j) {
                    Scalar dist = distance_squared(X.row(i), centroids_.row(j));
                    if (dist < min_dist) {
                        min_dist = dist;
                    }
                }
                distances(i) = min_dist;
            }

            // 基于距离平方的概率分布选择下一个中心
            Scalar total = distances.sum();
            std::uniform_real_distribution<> dist_prob(0, total);
            Scalar r = dist_prob(gen);
            Scalar cumulative = 0;

            for (int i = 0; i < n_samples; ++i) {
                cumulative += distances(i);
                if (cumulative >= r) {
                    centroids_.row(c) = X.row(i);
                    break;
                }
            }
        }
    }
    
    // 随机初始化聚类中心
    void initialize_centroids_random(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X) {
        const int n_samples = X.rows();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(0, n_samples - 1);

        for (int i = 0; i < n_clusters_; ++i) {
            centroids_.row(i) = X.row(dist(gen));
        }
    }

    // 计算惯性值
    void compute_inertia(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X) {
        inertia_ = 0.0;
        for (int i = 0; i < X.rows(); ++i) {
            inertia_ += distance_squared(X.row(i), centroids_.row(labels_(i)));
        }
    }

public:
    /**
     * @brief 构造函数
     * @param n_clusters 聚类数量
     * @param max_iter 最大迭代次数
     * @param tol 收敛阈值
     */
    KMeans(int n_clusters = 8, int max_iter = 300, double tol = 1e-4)
        : n_clusters_(n_clusters), max_iter_(max_iter), tol_(tol) {
        if (n_clusters <= 0) {
            throw std::invalid_argument("n_clusters must be positive");
        }
        if (max_iter <= 0) {
            throw std::invalid_argument("max_iter must be positive");
        }
        if (tol < 0) {
            throw std::invalid_argument("tol must be non-negative");
        }
    }

    /**
     * @brief 训练KMeans模型
     * @param X 输入数据，每行一个样本
     */
    void fit(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X) {
        const int n_samples = X.rows();
        const int n_features = X.cols();

        if (n_samples < n_clusters_) {
            throw std::invalid_argument("n_samples must be greater than n_clusters");
        }

        // 初始化聚类中心
        centroids_.resize(n_clusters_, n_features);
        initialize_centroids_kmeans_plus_plus(X);  // 使用k-means++初始化

        labels_.resize(n_samples);
        bool converged = false;
        int iter = 0;

        while (!converged && iter < max_iter_) {
            // 保存当前中心用于判断收敛
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> old_centroids = centroids_;

            // 分配样本到最近的聚类中心
            for (int i = 0; i < n_samples; ++i) {
                Scalar min_dist = std::numeric_limits<Scalar>::max();
                int best_cluster = 0;
                
                for (int j = 0; j < n_clusters_; ++j) {
                    Scalar dist = distance_squared(X.row(i), centroids_.row(j));
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_cluster = j;
                    }
                }
                labels_(i) = best_cluster;
            }

            // 计算新的聚类中心（每个簇的均值）
            Eigen::VectorXi cluster_sizes = Eigen::VectorXi::Zero(n_clusters_);
            centroids_.setZero();
            
            for (int i = 0; i < n_samples; ++i) {
                int cluster = labels_(i);
                centroids_.row(cluster) += X.row(i);
                cluster_sizes(cluster)++;
            }

            // 处理空簇
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dist(0, n_samples - 1);
            
            for (int j = 0; j < n_clusters_; ++j) {
                if (cluster_sizes(j) == 0) {
                    // 空簇时随机选择一个样本作为中心
                    centroids_.row(j) = X.row(dist(gen));
                } else {
                    centroids_.row(j) /= cluster_sizes(j);
                }
            }

            // 检查是否收敛
            Scalar centroid_shift = (centroids_ - old_centroids).norm();
            if (centroid_shift < tol_) {
                converged = true;
            }

            iter++;
        }

        // 计算惯性值
        compute_inertia(X);
    }

    /**
     * @brief 预测样本所属聚类
     * @param X 输入数据，每行一个样本
     * @return 每个样本的聚类标签
     */
    Eigen::VectorXi predict(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X) const {
        if (centroids_.size() == 0) {
            throw std::runtime_error("Model not fitted. Call fit first.");
        }

        const int n_samples = X.rows();
        Eigen::VectorXi labels(n_samples);
        
        for (int i = 0; i < n_samples; ++i) {
            Scalar min_dist = std::numeric_limits<Scalar>::max();
            int best_cluster = 0;
            
            for (int j = 0; j < n_clusters_; ++j) {
                Scalar dist = distance_squared(X.row(i), centroids_.row(j));
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            labels(i) = best_cluster;
        }
        
        return labels;
    }

    /**
     * @brief 拟合模型并预测
     * @param X 输入数据，每行一个样本
     * @return 每个样本的聚类标签
     */
    Eigen::VectorXi fit_predict(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X) {
        fit(X);
        return labels_;
    }

    /**
     * @brief 获取聚类中心
     * @return 聚类中心矩阵，每行一个中心
     */
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& cluster_centers() const {
        return centroids_;
    }

    /**
     * @brief 获取最后一次拟合的标签
     * @return 标签向量
     */
    const Eigen::VectorXi& labels() const {
        return labels_;
    }

    /**
     * @brief 获取惯性值（所有样本到其聚类中心的距离平方和）
     * @return 惯性值
     */
    double inertia() const {
        return inertia_;
    }
};

// namespace matrix_nn

#endif // KMEANS_H