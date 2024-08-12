from environment import Environment
import random


class Agent:
    def __init__(self):
        self.sum_reward = 0

    def step(self, env: Environment):
        print('-----------------------------------------')
        current_observation = env.get_location()
        print(f'Current Location: {current_observation}')

        actions = env.get_actions()

        random_action = random.choice(env.get_actions())
        print(f'Random Action is: {random_action}')

        reward = env.action(random_action)
        if reward != 0:
            print(f'Reward: {reward}')

        self.sum_reward += reward
