#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <ctime>
#include <map>

class MazeEnv {
public:
    // 动作定义
    enum Action {
        UP = 0,
        RIGHT = 1,
        DOWN = 2,
        LEFT = 3
    };

    // 构造函数
    MazeEnv(const std::vector<std::vector<int>>& maze_layout = {}) {
        // 迷宫布局: 0-通路, 1-墙壁, 2-起点, 3-终点
        if (maze_layout.empty()) {
            maze = {
                {1, 1, 1, 1, 1, 1, 1},
                {1, 2, 0, 0, 0, 0, 1},
                {1, 0, 1, 0, 1, 0, 1},
                {1, 0, 1, 0, 1, 0, 1},
                {1, 0, 1, 0, 1, 0, 1},
                {1, 0, 0, 0, 0, 3, 1},
                {1, 1, 1, 1, 1, 1, 1}
            };
        } else {
            maze = maze_layout;
        }

        rows = maze.size();
        cols = maze[0].size();
        max_steps = 1000;
        
        // 找到起点和终点
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (maze[i][j] == 2) {
                    start_pos = {i, j};
                    agent_pos = start_pos;
                } else if (maze[i][j] == 3) {
                    end_pos = {i, j};
                }
            }
        }
        
        step_count = 0;
    }

    // 重置环境
    std::pair<int, int> reset() {
        agent_pos = start_pos;
        step_count = 0;
        return {agent_pos.first, agent_pos.second};
    }

    // 执行动作
    std::tuple<std::pair<int, int>, float, bool, std::map<std::string, int>> step(Action action) {
        step_count++;
        std::pair<int, int> current_pos = agent_pos;

        // 根据动作更新位置
        switch (action) {
            case UP:    agent_pos.first--; break;
            case DOWN:  agent_pos.first++; break;
            case LEFT:  agent_pos.second--; break;
            case RIGHT: agent_pos.second++; break;
        }

        // 检查是否撞墙
        float reward = -1.0f;
        bool done = false;
        
        if (agent_pos.first < 0 || agent_pos.first >= rows || 
            agent_pos.second < 0 || agent_pos.second >= cols ||
            maze[agent_pos.first][agent_pos.second] == 1) {
            // 撞墙，回到原位置
            agent_pos = current_pos;
            reward = -10.0f;
        } else if (agent_pos == end_pos) {
            // 到达终点
            reward = 100.0f;
            done = true;
        }

        // 检查是否超过最大步数
        if (step_count >= max_steps) {
            done = true;
        }

        // 准备信息
        std::map<std::string, int> info;
        info["step_count"] = step_count;

        return {agent_pos, reward, done, info};
    }

    // 渲染迷宫
    void render() const {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (i == agent_pos.first && j == agent_pos.second) {
                    std::cout << "R ";  // 机器人
                } else if (maze[i][j] == 1) {
                    std::cout << "# ";  // 墙壁
                } else if (maze[i][j] == 2) {
                    std::cout << "S ";  // 起点
                } else if (maze[i][j] == 3) {
                    std::cout << "G ";  // 终点
                } else {
                    std::cout << ". ";  // 通路
                }
            }
            std::cout << std::endl;
        }
        std::cout << std::string(20, '-') << std::endl;
    }

private:
    std::vector<std::vector<int>> maze;
    std::pair<int, int> agent_pos;  // 机器人当前位置
    std::pair<int, int> start_pos;  // 起点位置
    std::pair<int, int> end_pos;    // 终点位置
    int rows, cols;                 // 迷宫尺寸
    int step_count;                 // 当前步数
    int max_steps;                  // 最大步数限制
};
