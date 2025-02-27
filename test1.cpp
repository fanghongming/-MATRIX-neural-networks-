#include <bits/stdc++.h>
#include <Eigen/Dense>
#include "Module.h"
#include "LinearLayer.h"
#include "Parameter.h"
#include "AdamOptimizer.h"
#include "Loss.h"
#include "Activation.h"
#include "NeuralNetwork.h"
#include "ModelIO.h"
#include "Dropout.h"
#include "L2Regularizer.h"
using namespace std;
class MsPacmanEnv {
    public:
        enum Action { UP, DOWN, LEFT, RIGHT, NOOP };
        static constexpr int ACTION_SPACE_SIZE = 5;
    
        struct Observation {
            std::array<int, 2> pacman_pos;
            std::vector<std::array<int, 2>> ghost_pos;
            std::vector<std::vector<bool>> dots_grid;
            std::vector<std::vector<bool>> walls_grid;
            int score;
            int lives;
        };
    
        MsPacmanEnv(int maze_width = 28, int maze_height = 31)
            : maze_w(maze_width), maze_h(maze_height) {
            reset();
        }
    
        Observation reset() {
            pacman = { maze_w / 2, maze_h - 2 };
    
            ghosts.clear();
            ghosts.push_back({5, 5});
            ghosts.push_back({maze_w - 6, 5});
            ghosts.push_back({maze_w / 2, 10});
    
            dots = std::vector<std::vector<bool>>(maze_h, std::vector<bool>(maze_w, true));
            walls = std::vector<std::vector<bool>>(maze_h, std::vector<bool>(maze_w, false));
            for (int i = 0; i < maze_h; ++i) {
                for (int j = 0; j < maze_w; ++j) {
                    if (i == 0 || i == maze_h - 1 || j == 0 || j == maze_w - 1)
                        walls[i][j] = true;
                }
            }
            for (int i = 10; i < 15 && i < maze_h - 1; ++i) {
                walls[i][maze_w / 2] = true;
            }
            for (int i = 3; i < 6 && i < maze_h - 1; ++i) {
                walls[i][maze_w / 3] = true;
            }
            for (int i = 9; i < 10 && i < maze_h - 1; ++i) {
                walls[i][static_cast<int>(maze_w / 1.5)] = true;
            }
            for (int i = 13; i < 20 && i < maze_w - 1; ++i) {
                walls[maze_h/3][i] = true;
            }
            for (int i = 5; i < 12 && i < maze_h - 1; ++i)
                walls[i][maze_w / 4] = true;
            for (int i = 8; i < 14 && i < maze_h - 1; ++i)
                walls[i][3 * maze_w / 4] = true;
            for (int j = 4; j < maze_w - 4; ++j)
                walls[maze_h / 3][j] = true;
            for (int j = 3; j < maze_w - 3; ++j)
                walls[2 * maze_h / 3 + 10][j] = true;
            for (int i = 5; i < 12 && i < maze_h - 1; ++i)
                walls[i+10][maze_w / 4] = true;
            for (int i = 8; i < 14 && i < maze_h - 1; ++i)
                walls[i+10][3 * maze_w / 4] = true;
            for (int j = 4; j < maze_w - 4; ++j)
                walls[maze_h / 3 + 10][j] = true;
            for (int i = 0; i < maze_h; ++i) {
                for (int j = 0; j < maze_w; ++j) {
                    if (walls[i][j])
                        dots[i][j] = false;
                }
            }
            score = 0;
            lives = 3;
            return get_observation();
        }
    
        std::tuple<Observation, int, bool> step(Action action) {
            move_pacman(action);
            move_ghosts();
    
            int reward = 0;
            bool done = false;
            if (dots[pacman[1]][pacman[0]]) {
                dots[pacman[1]][pacman[0]] = false;
                reward += 10;
                score += 10;
            }
            for (const auto& ghost : ghosts) {
                if (pacman == ghost) {
                    lives--;
                    reward -= 100;
                    reset_pacman();
                    break;
                }
            }
            if (lives <= 0)
                done = true;
            if (all_dots_eaten()) {
                reward += 500;
                done = true;
            }
            return { get_observation(), reward, done };
        }
    
        void render() const {
            //system("cls");
            for (int y = 0; y < maze_h; ++y) {
                for (int x = 0; x < maze_w; ++x) {
                    if (walls[y][x])
                        std::cout << "#";
                    else if (x == pacman[0] && y == pacman[1])
                        std::cout << "C";
                    else if (contains_ghost(x, y))
                        std::cout << "G";
                    else if (dots[y][x])
                        std::cout << ".";
                    else
                        std::cout << " ";
                }
                std::cout << "\n";
            }
            std::cout << "Score: " << score << "  Lives: " << lives << "\n";
        }
    
    private:
        int maze_w, maze_h;
        std::array<int, 2> pacman;
        std::vector<std::array<int, 2>> ghosts;
        std::vector<std::vector<bool>> dots;
        std::vector<std::vector<bool>> walls;
        int score;
        int lives;
    
        Observation get_observation() const {
            return { pacman, ghosts, dots, walls, score, lives };
        }
    
        void move_pacman(Action action) {
            std::array<int, 2> next = pacman;
            switch (action) {
                case UP:    if (pacman[1] > 1) next[1]--; break;
                case DOWN:  if (pacman[1] < maze_h - 2) next[1]++; break;
                case LEFT:  if (pacman[0] > 1) next[0]--; break;
                case RIGHT: if (pacman[0] < maze_w - 2) next[0]++; break;
                default: break;
            }
            if (!walls[next[1]][next[0]])
                pacman = next;
        }
    
        void move_ghosts() {
            static std::mt19937 rng(static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count()));
            std::uniform_int_distribution<int> dist(0, 3);
            for (auto& ghost : ghosts) {
                std::array<int, 2> next = ghost;
                int dir = dist(rng);
                switch (dir) {
                    case 0: if (ghost[1] > 1) next[1]--; break;
                    case 1: if (ghost[1] < maze_h - 2) next[1]++; break;
                    case 2: if (ghost[0] > 1) next[0]--; break;
                    case 3: if (ghost[0] < maze_w - 2) next[0]++; break;
                    default: break;
                }
                if (!walls[next[1]][next[0]])
                    ghost = next;
            }
        }
    
        bool all_dots_eaten() const {
            for (const auto& row : dots)
                if (std::any_of(row.begin(), row.end(), [](bool b) { return b; }))
                    return false;
            return true;
        }
    
        void reset_pacman() {
            pacman = { maze_w / 2, maze_h - 2 };
        }
    
        bool contains_ghost(int x, int y) const {
            return std::any_of(ghosts.begin(), ghosts.end(), [x, y](const std::array<int,2>& pos) {
                return pos[0] == x && pos[1] == y;
            });
        }
    };

struct Transition{
    Eigen::Matrix<double,1,2>state;
    int action;
    double reward;
    Eigen::Matrix<double,1,2>next_state;
    bool done;
};
class DQN{
    public:
    DQN(int state_dim,int action_dim,int hidden_dim,int memory_capacity=10000,
        int batch_size=32,double gamma=0.99,
        double epsilon=1.0,double epsilon_min=0.1,double epsilon_decay=0.995
        )
        :state_dim(state_dim),action_dim(action_dim),hidden_dim(hidden_dim),
        memory_capacity(memory_capacity),batch_size(batch_size),
        gamma(gamma),epsilon(epsilon),epsilon_min(epsilon_min),epsilon_decay(epsilon_decay)
        {
            q_networ.add_module()
        }
        private:
        
}
// signed main() {
//     // XOR 数据集
   
//     // fstream file("ccc.out",std::ios::in|std::ios::out|std::ios::trunc);
//     // file<<100<<"\n";
//     //fstream file("ccc.out",std::ios::in|std::ios::out|std::ios::trunc);
//     // 创建神经网络
//     NeuralNetwork<double> network;
//    // auto bias_optimizer=std::make_shared<AdamOptimizer<double,1,Eigen::Dynamic>>(Eigen::Matrix<double,1,Eigen::Dynamic>::Zero(1,4));
//    network.add_module(make_shared<LinearLayer<double>>(2, 4));
//     network.add_module(make_shared<ReLU<double>>());
//     network.add_module(make_shared<Dropout<double>>(0.5)); // 保留率50%
//     network.add_module(make_shared<LinearLayer<double>>(4, 1));
//     network.add_module(make_shared<Sigmoid<double>>());
//     load_model(network,"canshu.in");
//     MSELoss<double> loss;
//     network.set_training(1);
//     const int epochs = 0;
//     const double learning_rate = 0.1;
//     Eigen::MatrixXd input(1,2);//输出是行向量，出入还是行向量。
//     Eigen::MatrixXd target(1,1);
//     for (int epoch = 0; epoch < epochs; ++epoch) {
//         network.zero_grad();
//         int a=rand()%2;
//         int b=rand()%2;

        
//         input<<a,b;
      
//         target<<(a^b);
//         network.zero_grad();
// auto prediction = network.forward(input);             // 前向传播
// double loss_value = loss.forward(prediction, target);    // 计算损失
// auto grad_output = loss.backward(prediction, target);      // 得到损失梯度
// network.backward(grad_output);                           // 反向传播累加梯度
//         L2Regularizer<double> l2_reg(0.1);
// // 在这里累加 L2 正则化的梯度
//     for (auto& module : network.get_modules()) {
//         // 假设只有 LinearLayer 保存参数
//         if (auto linear = dynamic_cast<LinearLayer<double>*>(module.get())) {
//             // 将L2正则化梯度累加到原有梯度上
//             linear->weight.set_grad(linear->weight.get_grad() +
//                 l2_reg.grad(linear->weight.get_value()));
//             linear->bias.set_grad(linear->bias.get_grad() +
//                 l2_reg.grad(linear->bias.get_value()));
//             // 同时可以将正则化损失累加到总损失中（用于监控）
//             loss_value += l2_reg.loss(linear->weight.get_value());
//             loss_value += l2_reg.loss(linear->bias.get_value());
//         }
//     }

//     network.update();  // 更新参数
    
//         if (epoch % 1000 == 0) {
//             cout << "Epoch " << epoch << " - Loss: " << loss_value << endl;
//         }
//     }
//     network.set_training(0);
//     int cnt_loss=0;
//     for (int epoch = 0; epoch <10000; ++epoch) {
//         int a=rand()%2;
//         int b=rand()%2;

        
//         input<<a,b;
//         target<<(a^b);
//         auto prediction = network.forward(input);
//         auto loss_value = loss.forward(prediction, target);
//         auto grad_output = loss.backward(prediction, target);
//         if(prediction(0,0)>0.5)
//         {
//             if((a^b)==0)
//             {
//                 cnt_loss++;
//             }
//         }
//         else 
//         {
//             if((a^b)==1)
//             cnt_loss++;
//         }
//         network.zero_grad();
//         network.backward(grad_output);
//         network.update();

//         if (epoch % 1000 == 0) {
//             cout << "Epoch " << epoch << " - Loss: " << cnt_loss << endl;
//         }
//     }

//     auto final_prediction = network.forward(input);
//     cout << "Final Prediction: \n" << final_prediction << endl;
//     save_model(network,"canshu.in");
//     return 0;
// }
// template<typename Scalar>
// class NeuralNetwork : public Module<Scalar> {
// public:
//     void add_module(std::shared_ptr<Module<Scalar>> module) {
//         modules.push_back(module);
//     }

//     Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> forward(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input) override {
//         Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output = input;
//         for (auto& module : modules) {
//             output = module->forward(output);
//         }
//         return output;
//     }

//     Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> backward(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& grad_output) override {
//         Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> grad = grad_output;
//         for (auto it = modules.rbegin(); it != modules.rend(); ++it) {
//             grad = (*it)->backward(grad);
//         }
//         return grad;
//     }

//     void update() override {
//         for (auto& module : modules) {
//             module->update();
//         }
//     }

//     void zero_grad() override {
//         for (auto& module : modules) {
//             module->zero_grad();
//         }
//     }

// private:
//     std::vector<std::shared_ptr<Module<Scalar>>> modules;
// };
// //注意传入的是行向量，输出的也是行向量！
// //target也必须是行向量

// template<typename Scalar>
// class NeuralNetwork : public Module<Scalar> {
// public:
//     void add_module(std::shared_ptr<Module<Scalar>> module) {
//         modules.push_back(module);
//     }

//     Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> forward(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& input) override {
//         Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> output = input;
//         for (auto& module : modules) {
//             output = module->forward(output);
//         }
//         return output;
//     }

//     Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> backward(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& grad_output) override {
//         Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> grad = grad_output;
//         for (auto it = modules.rbegin(); it != modules.rend(); ++it) {
//             grad = (*it)->backward(grad);
//         }
//         return grad;
//     }

//     void update() override {
//         for (auto& module : modules) {
//             module->update();
//         }
//     }

//     void zero_grad() override {
//         for (auto& module : modules) {
//             module->zero_grad();
//         }
//     }

// private:
//     std::vector<std::shared_ptr<Module<Scalar>>> modules;
// };
// signed main() {
//     // 示例代码
//     Eigen::MatrixXd input(1, 3);
//     input << 1.0, 2.0, 3.0;

//     Eigen::MatrixXd target(1, 2);
//     target << 0.5, 1.5;

//     NeuralNetwork<double> network;
//     network.add_module(make_shared<LinearLayer<double>>(3, 2));
//     network.add_module(make_shared<ReLU<double>>());

//     MSELoss<double> loss;

//     const int epochs = 100;
//     Eigen::MatrixXd accumulated_grad_output;
//     accumulated_grad_output.setZero(1, 2); // 初始化累积梯度

//     for (int epoch = 0; epoch < epochs; ++epoch) {
//         auto prediction = network.forward(input);
//         auto loss_value = loss.forward(prediction, target);
//         auto grad_output = loss.backward(prediction, target);

//         accumulated_grad_output += grad_output; // 累积梯度

//         cout << "Epoch " << epoch + 1 << " - Loss: " << loss_value << endl;
//     }

//     accumulated_grad_output /= epochs; // 求平均梯度

//     network.backward(accumulated_grad_output); // 反向传播
//     network.update(); // 更新参数

//     auto final_prediction = network.forward(input);
//     cout << "Final Prediction: \n" << final_prediction << endl;

//     return 0;
// }

// signed main() {
//     // 示例代码
//     Eigen::MatrixXd input(1, 3);
//     input << 1.0, 2.0, 3.0;

//     Eigen::MatrixXd target(1, 2);
//     target << 0.5, 1.5;

//     LinearLayer<double> layer(3, 2);
//     MSELoss<double> loss;
//     ReLU<double> relu;

//     auto prediction = layer.forward(input);
//     auto activated_prediction = relu.forward(prediction);
//     auto loss_value = loss.forward(activated_prediction, target);
//     auto grad_output = loss.backward(activated_prediction, target);
//     auto grad_input = relu.backward(prediction, grad_output);

//     layer.backward(grad_input);
//     layer.update();

//     cout << "Prediction: \n" << prediction << endl;
//     cout << "Activated Prediction: \n" << activated_prediction << endl;
//     cout << "Loss: " << loss_value << endl;

//     return 0;
// }
// template<typename Scalar,int Rows,int Cols>
// struct Parameter{
//     Eigen::Matrix<Scalar,Rows,Cols>value;
//     Eigen::Matrix<Scalar,Rows,Cols>grad;

//     Parameter(const Eigen::Matrix<Scalar,Rows,Cols>&init_value)
//     :value(init_value)
//     {
//         grad=Eigen::Matrix<Scalar,Rows,Cols>::zero(init_value.rows(),init_value.cols());
//     }
//     Eigen::Matrix<Scalar,Rows,Cols>&get_value()
//     {
//         return value;
//     }
//     const Eigen::Matrix<Scalar,Rows,Cols>&get_value()const{
//         return value;
//     }
//     const Eigen::Matrix<Scalar,Rows,Cols>&get_grad()const
//     {
//         return grad;
//     }
//     void set_grad(const Eigen::Matrix<Scalar,Rows,Cols>&new_grad){
//         grad=new_grad;
//     }
//     void zero_grad()
//     {
//         grad.setZero();
//     }
// };
// template<typename Scalar,int Row,int Cols>
// struct Optimizer
// {
//     Parameter<Scalar,Rows,Cols>&param;
//     Optimizer (Parameter<Scalar,Rows,Cos>&P):Param(p)
//     {

//     }
//     virtual void step()=0;
// };
// template<typename Scalar,int Rows,int Cols>
// struct AdamOptimizer:Optimizer<Scalar,Rows,Cols>
// {
//     Scalar learning_rate;
//     Scalar beta1;
//     Scalar beta2;
//     Scalar epsilon;
//     int timestep;
//     Eigen::Matrix<scalar,Rows,Cols>m;
//     Eigen::Matrix<Scalar,Ros,Cols>v;

//     AdamOpimizer(Parameter<Scalar,Rows,Cols>&p,
//         Scalar lr=0.001,
//         Scalar b1=0.9,Scalar b2=0.999,Scalar eps=1e-8):
//         Optimizer<Scalar ,Rows,Cols>(p),learing_rate(lr),beta1(b1),beta2(b2),epsilon(eps),timstep(0)
//         {
//             m=Eigen::Matrix<Scalar,Rows,Cols>::Zero(p.get_grad().rows(),p.get_grad().cols());
//             v=Eigen::Matrix<Sclar,Rows,Cols>::Zero(p.get_grad().rows(),p.get_grad().cols());

//         }
//         void step()
//         {
//             timestep++;
//             auto &grad=this->param.get_grad();
//             auto &value=this->param.get_value();

//             m=beta1*m+(1-beta1)*grad;
//             v=beta2*v+(1-beta2)*grad.cwiseProduct(grad);
//             Eigen::Matrix<Scalar,Rows,Cols>m_hat=m/(1-std::pow(beta1,timestep));
//             Eigen::Matrix<Scalar,Rows,Cols>v_hat=v/(1-pow(beta2,timestep));

//             Eigen::Matrix<Scalar,Rows,Cols>epsilon_matrix(v_hat.rows(),v_hat.cols());
//             for(int i=0;i<v_hat.rows();i++)
//             {
//                 for(int j=0;j<v_hat.cols();j++)
//                 {
//                     epsilon_matrix(i,j)=epsilon;
//                 }
//             }
//             value-=learning_rate*m_hat.cwiseQuotient(v_hat.cwiseSqrt()+epsilon_matrix);
//         }
// };
// template<typename Scalar>
// struct  LinearLayer{
//     Parameter <Scalar,Eigen::Dynamic,Eigen::Dynamic>weight;
//     Parameter<Scalar,1,Eigen::Dynamic>bias;
//     AdamOptimizer<Scalar,Eigen::Dynamic,Eigen::Dynamic>weight_optimizer;
//     AdamOptimizer<Scalar,1,Eigen::Dynamic>bias_optimizer;

//     LinearLayer(int in_features,int out_features):
//     weight(Eigen::Matrix<Scalar,Eigen::Dyanmic,Eigen::Dynamic>Random(in_features,out_features)),
//     bias(Eigen::Matrix<Sclar,1,Eigen::Dynamic>::Random(1,out_features)),
//     weight_optimizer(weight),
//     bias_optimizer(bias)
//     {

//     }
//     Eigen::Matrix<Scalar,1,Eigen::Dynamic>forward(const Eigen::Matrix<Scalar,Eigne:Dynamic>&input)
//     {
//         return input*weight.get_value()+bias.get_value();
//     }
//     Eigen::Matrix<Scalar,1,Eigen::Dynamic>backward(const Eigen::Matrix<Scalar,1,Eigen::Dynamic>&input,
//     const Eigen::Matrix<Scalar,1,Eigen::Dynamic>&grad_ouput
//     )
//     {
//         Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>weight_grad=input.transpose()*grad_output;
//         weight.set_grad(weight_grad);

//         Eigen::Matrix<Scalar,1,Eigen::Dynamic>bias_grad=grad_output;
//         bias.set_grad(bias_grad);
//         Eigen::Matrix<Scalar,1,Eigen::Dynamic>grad_input=grad_output*weight.get_value().transpose();
//         return grad_input;
//     }
//     void update()
//     {
//         weight_optimizer.step();
//         bias_optimizer.step();
//     }
//     void zero_grad(){
//         weight.zero_grad();
//         bias.zero_grad();
//     }
// };
// template<class Scalar>
// struct Base{
//     Base()
//     {
        
//     }
//     void print()
//     {
//         cout<<"Base\n";
//     }
    
// };
// template<class Scalar>
// struct Derive:Base<Scalar>
// {
//     Derive()
//     {
//        // cout<<"Derive\n";
//     }
//     void print()
//     {
//         cout<<"Derive\n";
//     }
// };
// template<typename T>
// struct c
// {
//     c()
//     {
//         T b;
//        b.print();
//     }
// };

// signed main()
// {
//     c<Base<int>> a;
// }