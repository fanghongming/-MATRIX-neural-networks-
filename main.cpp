#include "BaseModel.h"
#include "SVM.h"
#include "NeuralNetwork.h"
#include "DecisionTree.h"
#include <Eigen/Dense>
#include "iostream"
#include <bits/stdc++.h>
#include"LinearLayer.h"
#include "Loss.h"
#include "Activation.h"
#include "Optimizer.h"
#include "ModelIO.h"
#include <iostream>
#include <vector>


#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <cassert>
#include "NeuralNetwork.h"
#include "LinearLayer.h"
#include "Activation.h"  // 假设包含ReLU、Sigmoid等激活函数
#include "Loss.h"
#include "ModelIO.h"     // 包含save_model和load_model函数

//浮点数比较辅助函数（处理精度问题）
int main() 
{
    // 生成示例数据（二分类问题）
        Eigen::MatrixXd X = Eigen::MatrixXd::Random(100, 2);  // 100样本，2特征
        Eigen::VectorXi y(100);
        for (int i = 0; i < 100; ++i) 
        {
            y(i) = (X(i, 0) + X(i, 1) > 0) ? 1 : -1;  // 简单标签
        }
        // 1. SVM模型
        BaseModel<double>*model1 = new SVC<double>(1.0, 1000);  
        model1->fit(X, y);
        std::cout << "SVM准确率: " << model1->score(X, y) << std::endl;
    // 在 main.cpp 中
        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>Y(100,1);
        for(int i=0;i<100;i++)
        {
            Y(i,0)=X(i,0)+X(i,1);
        }
    auto nn = new NeuralNetwork<double>();
    auto nn2=new NeuralNetwork<double>();
    // 添加含权重的线性层（输入2维，输出10维）
    nn->add_module(std::make_shared<LinearLayer<double>>(2,10));  
    nn2->add_module(std::make_shared<LinearLayer<double>>(2,10));

    nn->add_module(std::make_shared<ReLU<double>>());
    nn2->add_module(std::make_shared<ReLU<double>>());
    
    nn->add_module(std::make_shared<LinearLayer<double>>(10,1));
    nn2->add_module(std::make_shared<LinearLayer<double>>(10,1));
// 直接创建一个与第一个LinearLayer的weight维度匹配的Parameter
// 假设第一个LinearLayer的weight是 2x10 矩阵（in_features=2, out_features=10）
//Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> dummy_weight = Eigen::MatrixXd::Random(2, 10);
//auto param = std::make_shared<Parameter<double, Eigen::Dynamic, Eigen::Dynamic>>(dummy_weight);

// 将该Parameter传入AdamOptimizer

    nn->set_loss(
        std::make_shared<MSELoss<double>>()
    );
    nn2->set_loss(
        std::make_shared<MSELoss<double>>()
    );
    // // 2. 神经网络模型
    // auto nn = new NeuralNetwork<double>();
    // nn->add_module(std::make_shared<LinearLayer<double>>(2, 10));  // 输入层->隐藏层
    // nn->add_module(std::make_shared<ReLU<double>>());
    // nn->add_module(std::make_shared<LinearLayer<double>>(10, 2));  // 隐藏层->输出层
    // nn->set_loss_and_optimizer(
    //     std::make_shared<MSELoss<double>>(),
    //     std::make_shared<AdamOptimizer<double,Eigen::Dynamic,Eigen::Dynamic>>()  // 假设已实现优化器基类
    // );
    nn->set_max_iter(200);
    nn2->set_max_iter(200);
      // 多态：用基类指针指向子类
    nn->fit(X, Y);
    //std::cout << "神经网络准确率: " << model2->score(X, y) << std::endl;
    nn->save("a.w");
    nn2->load("a.w");
    auto ans=nn2->predict(X);
    //auto ans=nn->predict(X);
    std::cout<<Y.rows()<<" "<<Y.cols()<<" "<<ans.rows()<<" "<<ans.cols()<<"\n";
     for(int i=0;i<100;i++)
     {
         std::cout<<Y(i,0)<<" "<<ans(i,0)<<"\n";
     }
    //  for(int i=0;i<100;i++)
    //     {
    //         std::cout<<Y(i,0)<<" "<<Y(i,1)<<"\n";
    //     }
    // for(int i=0;i<100;i++)
    // {
    //     for(int j=0;j<2;j++)
    //     {
    //         std::cout<<Y(0,j)<<" ";
    //     }
    //     for(int j=0;j<2;j++)
    //     {
    //         std::cout<<ans(0,j)<<" ";
    //     }
    //     std::cout<<"\n";
    // }
    // for(int i=0;i<100;i++)
    // {
    //     for(int j=0;j<2;j++)
    //     {
    //         std::cout<<Y(i,j)<<" ";
    //     }
    //     for(int j=0;j<2;j++)
    //     std::cout<<ans(i,j)<<" ";
    //     std::cout<<"\n";
    // }
    // 3. 决策树模型
    BaseModel<double>* model3 = new DecisionTree<double>(5);
    model3->fit(X, y);


    std::cout << "决策树准确率: " << model3->score(X, y) << std::endl;
    // 释放资源
    delete model1;
    //delete model2;
    delete model3;

    return 0;
}

// template <typename Scalar = double>
// class KMeans {
// private:
//     int n_clusters_;          // 聚类数量
//     int max_iter_;            // 最大迭代次数
//     double tol_;              // 收敛阈值
//     Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> centroids_;  // 聚类中心
//     Eigen::VectorXi labels_;  // 最后一次拟合的标签
//     double inertia_;          // 惯性值（所有样本到其聚类中心的距离平方和）

//     // 计算欧氏距离的平方
//     Scalar distance_squared(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& a, 
//                            const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& b) const {
//         return (a - b).squaredNorm();
//     }

//     // 初始化聚类中心（k-means++算法）
//     void initialize_centroids_kmeans_plus_plus(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X) {
//        const int n_samples=X.rows();
//        const int n_features=X.cols();

//        std::random_device rd;

//        std::mt19937 gen(rd());

//        std::uniform_int_distribution<>dist(0,n_samples-1);

//        centroids_.row(0)=X.row(dist(gen));

//        for(int c=1;c<n_clusters_;++c)
//        {
//         Eigen::Vector<Scalar,Eigen::Dynamic>distances(n_samples);

//         for(int i=0;i<n_samples;i++)
//         {
//             Scalar min_dist=std::numeric_limits<Scalar>::max();

//             for(int j=0;j<c;j++)
//             {
//                 Scalar dist=distance_squared(X.rows(i),centroids_.row(j));

//                 if(dist<min_dist)
//                 {
//                     min_dist=dist;
//                 }
//             }
//             distances(i)=min_dist;
//         }
//         Scalar total=distances.sum();
//         std::uniform_real_distribution<>dist_prob(0,total);

//         Scalar cumulative=0;
//         for(int i=0;i<n_samples;i++)
//         {
//             cummulative+=distances(i);
//             if(cumlateive>=r)
//             {
//                 centroids_.row(c)=X.row(i);
//                 break;
//             }
//         }
//        }
//     }
    
//     // 随机初始化聚类中心
//     void initialize_centroids_random(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X) {
//         const int n_samples=X.rows();

//         std::random_device rd;
//         std::mt19937 gen(rd());

//         std::uniform_real_distribution<>dist(0,n_samples-1);

//         for(int i=0;i<n_clusters_;i++)
//         {
//             centroids_row(i)=X.row(dist(gen));
//         }
//     }


//     // 计算惯性值
//     void compute_inertia(const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamci>&X)
//     {
//         inertia_=0.0;
//         for(int i=0;i<X.rows();i++)
//         {
//             inertia_+=distance_squared(X.row(i),centroids_.row(labels_(i)));
//         }
//     }


// public:
//     /**
//      * @brief 构造函数
//      * @param n_clusters 聚类数量
//      * @param max_iter 最大迭代次数
//      * @param tol 收敛阈值
//      */
//     KMeans(int n_clusters = 8, int max_iter = 300, double tol = 1e-4)
//         : n_clusters_(n_clusters), max_iter_(max_iter), tol_(tol) {
//         if (n_clusters <= 0) {
//             throw std::invalid_argument("n_clusters must be positive");
//         }
//         if (max_iter <= 0) {
//             throw std::invalid_argument("max_iter must be positive");
//         }
//         if (tol < 0) {
//             throw std::invalid_argument("tol must be non-negative");
//         }
//     }

//     /**
//      * @brief 训练KMeans模型
//      * @param X 输入数据，每行一个样本
//      */
//     // void fit(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X) {
//     //     const int n_samples=X.rows();
//     //     const int n_features=X.cols();

//     //        if (n_samples < n_clusters_) {
//     //         throw std::invalid_argument("n_samples must be greater than n_clusters");
//     //     }

//     //     centroids_.resize(n_clusters_,n_features);
//     //     initializer_centroids_kmeans_plus_plus(X);

//     //     labels.resize(n_samples);

//     //     bool converged=false;

//     //     int iter=0;

//     //     while(!converged&&iter<max_iter)
//     //     {
//     //         Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>old_centroids=centroids_;
//     //         Eigen::Matrix <Scalar,Eigen::Dynamic,Eigen::Dynamic>old_centroids=centroids_;

//     //         for(int i=0;i<n_samp;es;i++)
//     //         {
//     //             Scalar min_dist=std::numeric_limits<Scalar>::max();

//     //             int best_cluster=0;

//     //             for(int j=0;j<n_clusters_;j++)
//     //             {
//     //                 Scalar dist=distance_squrared(X.rows(i),centroids_.row(j));

//     //                 if(dist<min_dist)
//     //                 {
//     //                     min_dist=dist;
//     //                     best_cluster=j;
//     //                 }
//     //             }
//     //             labels_(i)=best_cluster;
//     //         }
//     //         Eigen::VectorXi cluster_sizes=Eigen::VectorXi::Zero(n_clusters_);

//     //         centroids_.setZero();
//     //         for(int i=0;i<n_samples;i++)
//     //         {
//     //             int cluster=labels_(i);

//     //             centroids_.row(cluster)+=X.row(i);
//     //             cluster_sizes(cluster)++;
//     //         }
//     //         std::random_device rd;
//     //         std::mt19937 gen(rd());

//     //         std::uniform_real_distribution<>dist(0,n_samples-1);

//     //         for(int j=0;j<n_clusters_;j++)
//     //         {
//     //             if(cluster_sizes(j)==0)
//     //             {
//     //                 centroids_.rows(j)=X.row(dist(gen));

//     //             }
//     //             else 
//     //             centroids_row(j)/=cluster_sizes(j);
//     //         }
//     //         Scalar centroid_shift=(centroids_-old_centtroids).norm();

//     //         if(centroid_shift<tol_)
//     //         {
//     //             converged=true;
//     //         }
//     //         iter++;
//     //     }
//     //     compute_inertia(X);
//     // }
//     void fit(const Eigen::Matrix <Scalar,Eigen::Dynamic,Eigen::Dynamic>&x)
//     {
//         const int n_samples=X.rows();

//         const int n_features=X.cols();

//         if(n_samples<n_clusters_)
//         {
//             throw std::invalid_argument("n_samples must be greater than n_clusters");
//         }

//         centroids_.resize(n_clusters_,n_features);

//         initialize_centroids_kmeans_plus_plus(x);

//         labels_.resize(n_samples);

//         bool converged=false;

//         int iter=0;

//         while(!converged&&iter<max_iter_)
//         {
//             Eigen::Matrix<>Scalar,Eigen::Dynamic,Eigen::Dynamic>old_centroids=centroids_;

//             for(int i=0;i<n_samples;i++)
//             {
//                 Scalar min_dist=std::numeric_limits<Scalar>::max();

//                 int best_cluster=0;

//                 for(int j=0;j<n_clusters_;j++)
//                 {
//                     Scalar dist=distance_squared(X.row(i),centroids_.row(j));

//                     if(dist<min_dist)
//                     {
//                         min_dist=dist;
//                         best_cluster=j;
//                     }
//                 }
//                 labels_(i)=best_cluster;
//             }

//             Eigen::VectorXi cluster_sizes=Eigen::VectorXi::Zero(n_clusters_);

//             centroids_.setZero();

//             for(int i=0;i<n_samples;i++)
//             {
//                 int cluster=labels_(i);

//                 centroids_.row(cluster)+=X.row(i);

//                 cluster_sizes(cluster)++;
//             }

//             std::random_device rd;
//             std::mt19937 gen(rd());

//             std::uniform_int_distribution<>dist(0,n_samples-1);

//             for(int j=0;j<n_clusters_;j++)
//             {
//                 if(cluster_sizes(j)==0)
//                 {
//                     centroids_,row(j)=X.row(dist(gen));

//                 }
//                 else 
//                 {
//                     centroids_.row(j)/=cluster_sizes(j);
//                 }
//             }
//             Scalar centroids__shift(centroids_-old_centroids).norm();

//             if(centroids__shift<tol_)
//             {
//                 converged=true;
//             }
//             iter++;
//         }
//         compute_inertia(X);
//     }
//     Eigen::VectorXi predict(const Eigen::Matrix<Sclar,Eigen::Dynamic,EIgen::Dynamic>&X)const{
//         if(centroids_.size()==0)
//         {
//             throw std::runtime_error("Model not fitted .Call fit first");
//         }
//         const int n_samples=X.rows();

//         Eigen::VectorXi labels(n_samples);

//         for(int i=0;i<n_samples;i++)
//         {
//             Scalar min_dist=std::numeric_limits<Scalar>::max();

//             int best_cluster=0;

//             for(int j=0;j<n_clusters_;j++)
//             {
//                 Scalar dist=distance_squred(X.row(i),centroids_.row(j));

//                 if(dist<dist_dist)
//                 {
//                     min_dist=dist;

//                     best_cluster=j;
//                 }
//             }
//             labels(i)=best_cluster;
//         }
//         return labels;
//     }


//     /**
//      * @brief 预测样本所属聚类
//      * @param X 输入数据，每行一个样本
//      * @return 每个样本的聚类标签
//      */


//     /**
//      * @brief 拟合模型并预测
//      * @param X 输入数据，每行一个样本
//      * @return 每个样本的聚类标签
//      */
//     Eigen::VectorXi fit_predict(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X) {
//         fit(X);
//         return labels_;
//     }

//     /**
//      * @brief 获取聚类中心
//      * @return 聚类中心矩阵，每行一个中心
//      */
//     const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& cluster_centers() const {
//         return centroids_;
//     }

//     /**
//      * @brief 获取最后一次拟合的标签
//      * @return 标签向量
//      */
//     const Eigen::VectorXi& labels() const {
//         return labels_;
//     }

//     /**
//      * @brief 获取惯性值（所有样本到其聚类中心的距离平方和）
//      * @return 惯性值
//      */
//     double inertia() const {
//         return inertia_;
//     }
// };

// namespace matrix_nn

//#endif // KMEANS_H
// #include <iostream>
// #include <Eigen/Dense>
// #include "SVM.h"

// using namespace std;
// using namespace Eigen;

// // 生成线性可分的二分类测试数据
// void generate_linear_data(MatrixXd& X, VectorXi& y, int n_samples = 100) {
//     X = MatrixXd::Random(n_samples, 2);  // 2D特征
//     y = VectorXi(n_samples);

//     // 简单线性分隔：y=1 当 x0 + x1 > 0.2，否则 y=-1
//     for (int i = 0; i < n_samples; ++i) {
//         if (X(i, 0) + X(i, 1) > 0.2) {
//             y(i) = 1;
//         } else {
//             y(i) = -1;
//         }
//     }
// }

// int main() {
//     // 生成训练数据
//     MatrixXd X_train;
//     VectorXi y_train;
//     generate_linear_data(X_train, y_train, 200);

//     // 生成测试数据
//     MatrixXd X_test;
//     VectorXi y_test;
//     generate_linear_data(X_test, y_test, 50);

//     // 创建并训练SVM模型
//     SVC<> svm(1.0, 1000, 1e-3);  // C=1.0, 最大迭代1000次
//     svm.fit(X_train, y_train);

//     // 预测并评估
//     VectorXi y_pred = svm.predict(X_test);
//     double accuracy = svm.score(X_test, y_test);

//     // 输出结果
//     cout << "模型准确率: " << accuracy << endl;
//     cout << "支持向量数量: " << svm.support_vectors().rows() << endl;
//     cout << "权重向量w: " << svm.coef_().transpose() << endl;
//     cout << "偏置项b: " << svm.intercept_() << endl;

//     // 输出前10个样本的预测结果
//     cout << "\n前10个样本预测结果:" << endl;
//     for (int i = 0; i < 10; ++i) {
//         cout << "样本 " << i << ": 特征=(" << X_test(i,0) << "," << X_test(i,1) 
//              << "), 真实=" << y_test(i) << ", 预测=" << y_pred(i) << endl;
//     }

//     return 0;
// }
    
// #include <iostream>
// #include <Eigen/Dense>
// #include "DecisionTree.h"

// using namespace std;
// using namespace Eigen;

// // 生成分类测试数据
// void generate_test_data(MatrixXd& X, VectorXi& y, int n_samples = 100, int n_features = 2) {
//     X = MatrixXd::Random(n_samples, n_features);
//     y = VectorXi::Zero(n_samples);

//     // 简单分类规则：第一类x0 + x1 > 0，第二类x0 + x1 <= 0
//     for (int i = 0; i < n_samples; ++i) {
//         if (X(i, 0) + X(i, 1) > 0) {
//             y(i) = 1;
//         } else {
//             y(i) = 0;
//         }
//     }
// }

// int main() {
//     // 生成训练数据
//     MatrixXd X_train;
//     VectorXi y_train;
//     generate_test_data(X_train, y_train, 200, 2);

//     // 生成测试数据
//     MatrixXd X_test;
//     VectorXi y_test;
//     generate_test_data(X_test, y_test, 50, 2);

//     // 创建并训练决策树
//     DecisionTreeClassifier<> dt(5, 2, 1, 2);
//     dt.fit(X_train, y_train);

//     // 预测并评估
//     VectorXi y_pred = dt.predict(X_test);
//     double accuracy = dt.score(X_test, y_test);

//     // 输出结果
//     cout << "测试集预测结果（前10个）：" << endl;
//     for (int i = 0; i < 10; ++i) {
//         cout << "样本 " << i << ": 真实=" << y_test(i) << ", 预测=" << y_pred(i) << endl;
//     }
//     cout << "准确率: " << accuracy << endl;

//     return 0;
// }
    
// #include <iostream>
// #include <Eigen/Dense>
// #include "KMeans.h"

// using namespace std;
// using namespace Eigen;


// int main() {
//     // 生成三维高斯分布测试数据
//     const int n_samples = 1000;
//     const int n_features = 3;
//     const int n_clusters = 4;

//     MatrixXd data(n_samples, n_features);
//     random_device rd;
//     mt19937 gen(rd());
    
//     // 四个聚类中心
//     Vector3d centers[4] = {
//         Vector3d(0, 0, 0),
//         Vector3d(5, 5, 5),
//         Vector3d(10, 0, 10),
//         Vector3d(0, 10, 5)
//     };
    
//     // 生成数据
//     for (int i = 0; i < n_samples; ++i) {
//         int cluster = i % n_clusters;
//         normal_distribution<> dist_x(centers[cluster](0), 1.0);
//         normal_distribution<> dist_y(centers[cluster](1), 1.0);
//         normal_distribution<> dist_z(centers[cluster](2), 1.0);
        
//         data(i, 0) = dist_x(gen);
//         data(i, 1) = dist_y(gen);
//         data(i, 2) = dist_z(gen);
//     }

//     // 训练KMeans模型
//     KMeans<> kmeans(n_clusters, 300, 1e-4);
//     kmeans.fit(data);

//     // 输出结果
//     cout << "聚类中心:\n" << kmeans.cluster_centers() << endl;
//     cout << "前10个样本的标签:\n" << kmeans.labels().head(10) << endl;
//     cout << "惯性值: " << kmeans.inertia() << endl;

//     // 预测新样本
//     MatrixXd new_samples(2, n_features);
//     new_samples << 1, 1, 1,
//                    6, 6, 6;
//     VectorXi new_labels = kmeans.predict(new_samples);
//     cout << "新样本的预测标签: " << new_labels.transpose() << endl;

//     return 0;
// }