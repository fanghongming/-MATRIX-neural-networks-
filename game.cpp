// C++
#include <iostream>
#include <vector>
#include <deque>
#include <random>
#include <chrono>
#include <thread>
#include <algorithm>
#include <cstdlib>
#include <Windows.h>
#include<bits/stdc++.h>
#include "ModelIO.h"
#include <Eigen/Dense>
#include "NeuralNetwork.h"   // 用于构建 Q 网络 [NeuralNetwork.h](NeuralNetwork.h)
#include "LinearLayer.h"       // [LinearLayer.h](LinearLayer.h)
#include "Activation.h"        // [Activation.h](Activation.h)
#include "Loss.h"              // [Loss.h](Loss.h)

// -------------- MsPacmanEnv --------------
// 直接拷贝自 [game.cpp](game.cpp)
struct MsPacmanEnv {
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

    MsPacmanEnv(int maze_width = 15, int maze_height =15)
        : maze_w(maze_width), maze_h(maze_height) {
        reset();
    }

    Observation reset() {
        pacman = { maze_w / 2, maze_h - 2 };

        ghosts.clear();

        dots = std::vector<std::vector<bool>>(maze_h, std::vector<bool>(maze_w, true));
        walls = std::vector<std::vector<bool>>(maze_h, std::vector<bool>(maze_w, false));
        for (int i = 0; i < maze_h; ++i) {
            for (int j = 0; j < maze_w; ++j) {
                if (i == 0 || i == maze_h - 1 || j == 0 || j == maze_w - 1)
                    walls[i][j] = true;
            }
        }
       for(int i=0;i<maze_h;i++)
       {
            for(int j=0;j<maze_w;j++)
            {
                int t=rand()%1000;
                if(t<=100)
                {
                    walls[i][j]=true;
                }
            }
       }
       for(int i=1;i<=20;i++)
        {
            int xx=rand()%(maze_w-1)+1;
            int yy=rand()%(maze_h-1)+1;
            if(walls[xx][yy])
            {
                continue;
            }
            ghosts.push_back({xx,yy});
        }
        for (int i = 0; i < maze_h; ++i) {
            for (int j = 0; j < maze_w; ++j) {
                if (walls[i][j])
                    dots[i][j] = false;
            }
        }
        score = 0;
        lives = 1;
        return get_observation();
    }

    std::tuple<Observation, int, bool> step(Action action) {
        move_pacman(action);
        move_ghosts();

        int reward = 0;
        bool done = false;
        if (dots[pacman[1]][pacman[0]]) {
            dots[pacman[1]][pacman[0]] = false;
            reward += 50;
            score += 10;
        }
        else
        {
            reward-=100;
        }
        for (const auto& ghost : ghosts) {
            if (pacman == ghost) {
                //std::cout<<ghost[0]<<" "<<ghost[1]<<" "<<pacman[0]<<" "<<pacman[1]<<"\n";
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

// -------------- DQN 实现 --------------
struct Transition {
    Eigen::Matrix<double, 1, Eigen::Dynamic> state;     // 使用吃豆人位置作为状态：二维向量
    int action;
    double reward;
    Eigen::Matrix<double, 1, Eigen::Dynamic> next_state;
    bool done;
};

class DQN {
public:
    DQN(int state_dim, int action_dim, int hidden_dim, int memory_capacity = 1000,
        int batch_size = 32, double gamma = 0.99,
        double epsilon = 1.0, double epsilon_min = 0.1, double epsilon_decay = 0.995)
        : state_dim(state_dim), action_dim(action_dim), hidden_dim(hidden_dim),
          memory_capacity(memory_capacity), batch_size(batch_size),
          gamma(gamma), epsilon(epsilon), epsilon_min(epsilon_min), epsilon_decay(epsilon_decay) {
        // 构建 Q 网络和目标网络
        q_network.add_module(std::make_shared<LinearLayer<double>>(state_dim, hidden_dim));
        q_network.add_module(std::make_shared<ReLU<double>>());
        q_network.add_module(std::make_shared<LinearLayer<double>>(hidden_dim, action_dim));

        target_network = q_network; // 初始目标网络与 Q 网络一致
    }

    // 利用 epsilon-greedy 策略选择动作
    int select_action(const Eigen::Matrix<double, 1, Eigen::Dynamic>& state) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(rng) < epsilon) {
            std::uniform_int_distribution<int> action_dist(0, action_dim - 1);
            return action_dist(rng);
        } else {
            Eigen::MatrixXd q_values = q_network.forward(state);
            int best_action;
            q_values.row(0).maxCoeff(&best_action);
            return best_action;
        }
    }

    // 将新体验存储到经验回放中
    void store_transition(const Transition& trans) {
        if (memory.size() >= memory_capacity)
            memory.pop_front();
        memory.push_back(trans);
    }

    // 进行一次网络更新（从经验中抽样 mini-batch)
    void update() {
         if (memory.size() < batch_size)
            return;
        //std::cout<<"now update\n";
        std::uniform_int_distribution<int> index_dist(0, memory.size() - 1);
        std::vector<Transition> batch;
        for (int i = 0; i < batch_size; ++i) {
            int idx = index_dist(rng);
            batch.push_back(memory[idx]);
        }

        // 构造当前 Q 输出与目标 Q 的矩阵（逐样本更新）
        double total_loss = 0.0;
        MSELoss<double> mse_loss;
        for (const auto& trans : batch) {
            // 当前 Q(s)
            Eigen::Matrix<double, 1, Eigen::Dynamic> state = trans.state; // (1,2)
            //std::cout<<state.size()<<" "<<state_dim<<"\n";
            Eigen::MatrixXd q_eval = q_network.forward(state); // (1, action_dim)
           // 目标值
            double target_val = trans.reward;
            if (!trans.done) {
                Eigen::MatrixXd q_next = target_network.forward(trans.next_state);
                double max_q;
                q_next.row(0).maxCoeff(&max_q);
                target_val += gamma * max_q;
            }
            // 构造目标向量：与 q_eval 相同，替换动作值
            Eigen::MatrixXd q_target = q_eval;
            q_target(0, trans.action) = target_val;

            // 计算误差并进行反向传播
            double loss = mse_loss.forward(q_eval, q_target);
            total_loss += loss;
            Eigen::MatrixXd grad = mse_loss.backward(q_eval, q_target);
            q_network.zero_grad();
            q_network.backward(grad);
            q_network.update();
        }
        // 衰减 epsilon
        if (epsilon > epsilon_min)
            epsilon *= epsilon_decay;
    }
    void save(std::string name)
    {
        save_model(q_network,name);
    }   
    void load(std::string name)
    {
        load_model(q_network,name);
        load_model(target_network,name);

    }
    // 将当前 Q 网络参数同步到目标网络
    void update_target() {
        target_network = q_network;
    }

private:
    int state_dim;
    int action_dim;
    int hidden_dim;
    int memory_capacity;
    int batch_size;
    double gamma;
    double epsilon;
    double epsilon_min;
    double epsilon_decay;
    std::deque<Transition> memory;
    NeuralNetwork<double> q_network;
    NeuralNetwork<double> target_network;
    std::mt19937 rng { static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count()) };
};

// 简单的状态提取函数：将 MsPacmanEnv 的 Observation 转为 (1,2) 行向量，取吃豆人位置
Eigen::Matrix<double, 1, Eigen::Dynamic> extract_state(const MsPacmanEnv::Observation& obs) {
    // 设定地图尺寸
    int H = obs.walls_grid.size();
    int W = (H > 0) ? obs.walls_grid[0].size() : 0;
    // 状态向量包含：吃豆人位置（2个） + dots（H*W） + walls（H*W）
    int total_dim = 2 + H * W * 2;
    Eigen::Matrix<double, 1, Eigen::Dynamic> state(total_dim);
    
    // 填入吃豆人位置（归一化可根据需要调整，这里直接使用整数）
    state(0, 0) = obs.pacman_pos[0];
    state(0, 1) = obs.pacman_pos[1];
    
    int idx = 2;
    // 拉平 dots_grid（存在豆子记为1.0，否则为0.0）
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            state(0, idx++) = obs.dots_grid[i][j] ? 1.0 : 0.0;
        }
    }
    // 拉平 walls_grid（墙体记为1.0，否则为0.0）
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            state(0, idx++) = obs.walls_grid[i][j] ? 1.0 : 0.0;
        }
    }
    
    return state;
}

int main() {
    MsPacmanEnv env;
    const int state_dim = env.maze_h*env.maze_w*2+2;    // 仅使用吃豆人位置作为状态
    const int action_dim = MsPacmanEnv::ACTION_SPACE_SIZE;
    const int hidden_dim = 500;
    DQN agent(state_dim, action_dim, hidden_dim);
    agent.load("agent.w");
    const int num_episodes = 500;
    const int target_update_freq = 5;
    
    for (int episode = 0; episode < num_episodes; ++episode) {
        auto obs = env.reset();
        Eigen::Matrix<double, 1, Eigen::Dynamic> state = extract_state(obs);
        double episode_reward = 0.0;
        bool done = false;
        //std::cout<<state.rows()<<" "<<state.cols()<<"\n";
        //std::cout<<state_dim<<"\n";
        while (!done) {
            //env.render();
            int action = agent.select_action(state);
            auto [next_obs, reward, done_flag] = env.step(static_cast<MsPacmanEnv::Action>(action));
            Eigen::Matrix<double, 1, Eigen::Dynamic> next_state = extract_state(next_obs);
            Transition trans { state, action, reward, next_state, done_flag };
            agent.store_transition(trans);
            agent.update();
            state = next_state;
            episode_reward += reward;
           
            //std::cout << "Reward: " << reward << "\n";
            //Sleep(1000);
            done = done_flag;
        }
        if (episode % target_update_freq == 0)
            agent.update_target();
        std::cout << "Episode " << episode << " Total Reward: " << episode_reward << "\n";
        Sleep(1000);
    }
    std::cout << "Training finished.\n";
    agent.save("agent.w");
    return 0;
}