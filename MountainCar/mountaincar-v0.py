import gym
import numpy as np
import pickle
import os

class GymRunner:
    def __init__(self, env_name, iter_max = 10000):
        self.env = gym.make(env_name)
        self.iter_max = iter_max
        self.n_state = 40

    def action_size(self):
        return self.env.action_space.n

    def train(self, agent, eposides):
        self.run(agent, eposides)

    def run(self, agent, eposides):
        for eposide in range(eposides):
            state = self.obs_to_state(self.env.reset())
            total_rewards = 0

            for i in range(self.iter_max):
                action = agent.select_action(state)
                next_obs, reward, done, _ = self.env.step(action)
                next_state = self.obs_to_state(next_obs)

                agent.update_q_table(state, next_state, action, reward, eposide)
                total_rewards += reward

                if done:
                    break

                state = next_state
            if eposide % 100 == 0:
                print('Iteration #%d -- Total reward = %d.' %(eposide+1, total_rewards))
    def test(self, policy):
        obs = self.env.reset()
        total_rewards = 0
        for i in range(self.iter_max):
            self.env.render()

            a,b = self.obs_to_state(obs)
            action = policy[a][b]

            obs,reward,done,_ = self.env.step(action)
            total_rewards += reward
            if done:
                break
            
        print("total rewards:{}".format(total_rewards))


    def obs_to_state(self, obs):
        env_low = self.env.observation_space.low
        env_high = self.env.observation_space.high

        env_dx = (env_high - env_low) / self.n_state

        a = int((obs[0] - env_low[0]) / env_dx[0])
        b = int((obs[1] - env_low[1]) / env_dx[1])

        return a, b


class QLearningAgent:
    def __init__(self, action_size, n_state):
        self.epsilon = 0.02
        self.learning_rate = 1.0
        self.min_lr = 0.003
        self.gamma = 1.0
        
        self.action_size = action_size
        self.q_table = read_dataset("q_table.p")
        if self.q_table is None:
            self.q_table = np.zeros((n_state, n_state, 3))
        

    def select_action(self, state):
        a, b = state
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            logits = self.q_table[a][b]
            logits_exp = np.exp(logits)
            probs = logits_exp / np.sum(logits_exp)
            return np.random.choice(self.action_size, p = probs)


    def update_q_table(self, cur_state, next_state, action, reward, i):
        a, b = cur_state
        a_,b_ = next_state
        #self.q_table[a][b][action] += self.learning_rate * (reward + self.gamma * np.max(self.q_table[a_][b_]) - self.q_table[a][b][action])
        self.q_table[a][b][action] = self.q_table[a][b][action] + self.learning_rate * (reward + self.gamma * np.max(self.q_table[a_][b_]) - self.q_table[a][b][action])
        self.learning_rate = max(self.min_lr, self.learning_rate * (0.85 ** (i // 100)))

    def test_policy(self):
        return np.argmax(self.q_table, axis=2)
    def close_save(self):
        save_dataset("q_table.p", self.q_table)


def save_dataset(filename, data):
    pickle.dump(data, open(filename, 'wb+'))

def read_dataset(filename):
    if os.path.isfile(filename):
        return pickle.load(open(filename, 'rb'))
    return None

if __name__ == "__main__":
    mountain_car = GymRunner("MountainCar-v0")
    agent = QLearningAgent(mountain_car.action_size(), mountain_car.n_state)
    mountain_car.train(agent, 10000)
    agent.close_save()
    mountain_car.test(agent.test_policy())






