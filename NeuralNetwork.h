#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include "Module.h"
template<typename Scalar>
class NeuralNetwork : public Module<Scalar> {
public:
    void add_module(std::shared_ptr<Module<Scalar>> module) {
        modules.push_back(module);
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> forward(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input) override {
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output = input;
        for (auto& module : modules) {
            output = module->forward(output);
        }
        return output;
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> backward(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& grad_output) override {
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> grad = grad_output;
        for (auto it = modules.rbegin(); it != modules.rend(); ++it) {
            grad = (*it)->backward(grad);
        }
        return grad;
    }

    void update() override {
        for (auto& module : modules) {
            module->update();
        }
    }

    void zero_grad() override {
        for (auto& module : modules) {
            module->zero_grad();
        }
    }

private:
    std::vector<std::shared_ptr<Module<Scalar>>> modules;
};
//注意传入的是行向量，输出的也是行向量！
//target也必须是行向量
//调试时，直接传入到函数内部调试。
#endif 