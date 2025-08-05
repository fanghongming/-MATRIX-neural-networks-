// NeuralNetwork.h（改造后）
#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "BaseModel.h"
#include "Module.h"
#include "Loss.h"
#include "Optimizer.h"
#include "LinearLayer.h"
// template<class Scalar>
// class NeuralNetwork:public Module<Scalar>
// {
//     public:
//         void add_module(std::shared_ptr<Module<Scalar>>module)
//         {
//             modules.push_back(module);
//         }
//         private:
//         std::vector<std::shared_ptr<Module<Scalar>>>modules;
//         Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>backward(const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>&grad_output)
//         {
//             Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>grad=grad_output;

//             for(auto it=modules.rbegin();it!=modules.rend();++it)
//             {
//                 grad=(*it)->backward(grad);
//             }
//             return grad;
//         }
//         void update()override
//         {
//             for(auto &module:modules)
//             {
//                 module->update();
//             }
//         }
//         void zero_grad()override
//         {
//             for(auto &module:modules)
//             {
//                 module->zero_grad();
//             }
//         }
//         std::vector<std::shared_ptr<module<Scalar>>>&get_modules()
//         {
//             return modules;
//         }
//         void set_training(bool training)override{
//             for(auto &module:modules)
//             {
//                 module->set_training(training);
//             }
//         }
//         private:

//         std::vector<std::shared_ptr<Moudle<Scalar>>>modules;
// };
template<typename Scalar>
class NeuralNetwork : public Module<Scalar> {  // 多继承：BaseModel+Module
public:
    // 新增：绑定损失函数和优化器
    void set_loss(
        std::shared_ptr<Loss<Scalar>> loss_fn  // 假设Optimizer为所有优化器的基类
    )
    {
        loss_fn_ = loss_fn;
    }

    // 添加网络层（原有方法）
    void add_module(std::shared_ptr<Module<Scalar>> module) {   
        modules.push_back(module);
    }
    bool is_training_=true;
    // 前向传播（实现Module接口）
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> forward
    (
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input
    )  override {
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output = input;
        for (auto& module : modules) 
        {
            output = module->forward(output);
        }
        return output;
    }

    // 反向传播（实现Module接口）
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> backward(
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& grad_output
    ) override {
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> grad = grad_output;
        for (auto it = modules.rbegin(); it != modules.rend(); ++it) {
            grad = (*it)->backward(grad);
        }
        return grad;
    }
// 模式切换：训练模式（默认空实现，需状态管理的模型重写）
    void train() {
        is_training_ = true;
    }

    // 模式切换：评估模式（默认空实现，需状态管理的模型重写）
     void eval() {
        is_training_ = false;
    }

    // 获取当前模式
    bool is_training() const {
        return is_training_;
    }
    
    // 重写BaseModel的fit方法（实现训练逻辑）
    void fit(
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& X,
        const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> &y)  
    {
        
        if (!loss_fn_ ) {
            throw std::runtime_error("请先调用set_loss设置损失函数");
        }
        if (X.rows() != y.rows())
        {
            throw std::invalid_argument("样本数与标签数不匹配");
        }

        // 训练模式下启用dropout等层
        this->train();  // 调用BaseModel的train()切换模式
        for (auto& module : modules) {
            module->set_training(this->is_training());
        }

        // 简单训练循环（可扩展为批次训练）
        for (int iter = 0; iter < max_iter_; ++iter) {
            for(int i=0;i<X.rows();i++)
            {
                Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>input=X.row(i);
                Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>output=y.row(i);
                //获取每一行的这个元素，每一行作为一个数据。
                auto y_pred=forward(input);
                Scalar loss=loss_fn_->forward(y_pred,output);
                zero_grad();
                auto grad=loss_fn_->backward(y_pred,output);
                //获取最后面的梯度。
                backward(grad);//反向传播。
                update();//更新操作。
            }
            // 反向传播

            // 优化器更新参数
            // optimizer_->step();
        }
    }
    std::vector<std::shared_ptr<Parameter<Scalar, Eigen::Dynamic, Eigen::Dynamic>>> get_parameters() {
        std::vector<std::shared_ptr<Parameter<Scalar, Eigen::Dynamic, Eigen::Dynamic>>> params;
        for (auto& module : modules) {
            // 动态判断模块是否为 LinearLayer（含可训练参数）
            if (auto linear = dynamic_cast<LinearLayer<Scalar>*>(module.get())) {
                params.push_back(std::make_shared<Parameter<Scalar, Eigen::Dynamic, Eigen::Dynamic>>(linear->weight));
                params.push_back(std::make_shared<Parameter<Scalar, Eigen::Dynamic, Eigen::Dynamic>>(linear->bias));
            }
        }
        return params;
    }
    Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>predict(
        const Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>&X
    )
    {
        this->eval();
        Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>ans(X.rows(),forward(X.row(0)).cols());
        for(int i=0;i<X.rows();i++)
        {
            auto output=forward(X.row(i));

            for(int j=0;j<ans.cols();j++)
            {
                ans(i,j)=output(0,j);
            }
        }
        return ans;
    }
    // 重写BaseModel的predict方法
    // Eigen::VectorXi predict(
    //     const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynautoamic>& X
    // )  {
    
    //     this->eval();  // 评估模式
    //     auto output = forward(X);  // 前向传播获取输出
    //     Eigen::VectorXi y_pred(X.rows());
    //     // 假设输出为概率分布，取最大值索引作为预测类别
    //     for (int i = 0; i < X.rows(); ++i) {
    //         output.row(i).maxCoeff(&y_pred(i));
    //     }
    //     return y_pred;
    // }
    // 其他原有方法（update、zero_grad等）保持不变
    void update() override {
        for (auto& module : modules) module->update();
    }
    void zero_grad() override {
        for (auto& module : modules) module->zero_grad();
    }
    void set_training(bool training) override {
        for (auto& module : modules) module->set_training(training);
    }
    void save(const std::string&filename)const 
    {
        std::fstream file(filename,std::ios::out|std::ios::trunc);
        if(!file.is_open())
        {
            throw std::runtime_error("无法打开文件用于保存模型"+filename);
        }
        size_t num_modules=modules.size();

        file<<num_modules<<"\n";

        for(const auto &module:modules)
        {
            module->save(file);
        }
        file.close();
    }
    void load(const std::string&filename)
    {
        std::fstream file(filename,std::ios::in);

        if(!file.is_open())
        {
            throw std::runtime_error("无法打开文件用于加载模型"+filename);
        }
        size_t num_modules;
        file>>num_modules;
        if(num_modules!=modules.size())
        {
            throw std::runtime_error("加载的模型模块模块数量和当前网络不匹配");
        }

        for(auto &module:modules)
        {
            module->load(file);
        }
        file.close();
    }
    // 新增：设置最大迭代次数
    void set_max_iter(int max_iter) { max_iter_ = max_iter; }

private:
    std::vector<std::shared_ptr<Module<Scalar>>> modules;
    std::shared_ptr<Loss<Scalar>> loss_fn_;    // 损失函数
    std::shared_ptr<OptimizerBase> optimizer_;  // 优化器
    int max_iter_ = 1000;  // 训练迭代次数

    // 辅助函数：将标签向量转为矩阵（适配损失函数输入）
    // Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> label_to_matrix(const Eigen::VectorXi& y) const {
    //     int n_classes = y.maxCoeff() + 1;  // 假设标签从0开始
    //     Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> y_mat(y.size(), n_classes);
    //     y_mat.setZero();
    //     for (int i = 0; i < y.size(); ++i) {
    //         y_mat(i, y(i)) = 1;  // 转为one-hot编码
    //     }
    //     return y_mat;
    // }
};
#endif // NEURALNETWORK_H