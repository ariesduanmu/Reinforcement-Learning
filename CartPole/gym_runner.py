import gym
from gym import wrappers

class GymRunner:
    def __init__(self, env_name, monitor_dir, iter_max = 10000):
        self.monitor_dir = monitor_dir
        self.iter_max = iter_max

        self.env = gym.make(env_name)
        self.env = wrappers.Monitor(self.env, monitor_dir, force = True)

    def calc_reward(self, state, action, gym_reward, next_state, done):
        return gym_reward

    def train(self, agent, episodes):
        self.run(agent, episodes, True)

    def run(self, agent, episodes, do_train = False):
        for episode in range(episodes):
            state = self.env.reset().reshape(1, self.env.observation_space.shape[0])
            total_reward = 0

            for i in range(self.iter_max):
                action = agent.select_action(state, do_train)
                next_state, reward, done, _ = self.env.step(action)
                next_state = next_state.reshape(1, self.env.observation_space.shape[0])

                reward = self.calc_reward(state, action, reward, next_state, done)

                if do_train:
                    agent.record(state, action, reward, next_state, done)

                total_reward += reward
                state = next_state
                if done:
                    break

            if do_train:
                agent.replay()

            print("episode: {}/{} | score: {} | e: {:.3f}".format(
                episode + 1, episodes, total_reward, agent.epsilon))


    def close_and_upload(self, api_key):
        self.env.close()
        #gym.upload(self.monitor_dir, api_key = api_key)