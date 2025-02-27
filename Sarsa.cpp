#include <iostream>
#include <tuple>
#include <random>
#include <Eigen/Dense>

using namespace std;
using Eigen::MatrixXd;

// 一个简单的链式环境，状态 0 到 5，其中状态 0 和 5 设为终止状态
struct Environment {
    int num_states;
    int start_state;
    int terminal_state_left;
    int terminal_state_right;
    
    Environment() : num_states(6), start_state(2), terminal_state_left(0), terminal_state_right(5) { }
    
    // step 返回 (next_state, reward, done)
    tuple<int, double, bool> step(int state, int action) {
        // action 0 表示向左移动，1 表示向右移动
        int next_state = state;
        if (action == 1) {
            next_state = state + 1;
        } else {
            next_state = state - 1;
        }
        
        // 奖励设定：到达右端终止状态获得奖励 1，其它0
        double reward = (next_state == terminal_state_right) ? 1.0 : 0.0;
        bool done = (next_state == terminal_state_left || next_state == terminal_state_right);
        return make_tuple(next_state, reward, done);
    }
};

int main() {
    // 参数设置
    const int num_states = 6;
    const int num_actions = 2; // 0: left, 1: right
    const double alpha = 0.1;  // 学习率
    const double gamma = 0.99; // 折扣因子
    const double epsilon = 0.1; // 探索率
    const int num_episodes = 1000;
    
    // 初始化 Q 表，行数为状态数，列数为动作数
    MatrixXd Q = MatrixXd::Zero(num_states, num_actions);
    
    // 随机数生成器
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);
    uniform_int_distribution<> action_dis(0, num_actions - 1);
    
    // epsilon-greedy 策略函数
    auto epsilon_greedy = [&](int state) -> int {
        if (dis(gen) < epsilon) {
            return action_dis(gen);
        } else {
            if (Q(state, 0) == Q(state, 1))
                return action_dis(gen);
            return (Q(state, 0) >= Q(state, 1)) ? 0 : 1;
        }
    };
    
    Environment env;
    
    // 每个 episode 的循环
    for (int episode = 0; episode < num_episodes; episode++) {
        int state = env.start_state;
        int action = epsilon_greedy(state);
        bool done = false;
        
        // SARSA 算法：在一个 trajectory 内更新 Q 表
        while (!done) {
            auto [next_state, reward, done_flag] = env.step(state, action);
            done = done_flag;
            int next_action = epsilon_greedy(next_state);
            
            // TD 目标和误差
            double td_target = reward;
            if (!done) {
                td_target += gamma * Q(next_state, next_action);
            }
            double td_error = td_target - Q(state, action);
            Q(state, action) += alpha * td_error;
            
            state = next_state;
            action = next_action;
        }
        
        if (episode % 100 == 0) {
            cout << "Episode " << episode << " Q table:\n" << Q << "\n";
        }
    }
    
    cout << "Final Q Table:\n" << Q << endl;
    return 0;
}