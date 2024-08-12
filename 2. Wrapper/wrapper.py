import gym
from typing import TypeVar
import random

Action = TypeVar('Action')


class SlipperyScenario(gym.ActionWrapper):
    def __init__(self, env, epsilon):
        super(SlipperyScenario, self).__init__(env)
        self.epsilon = epsilon

    # We override this method, which is present in ActionWrapper Class.
    def action(self, action: Action):
        if random.Random < self.epsilon:
            print('Random!')
            while action != self.env.action_space.sample():
                return self.env.action_space.sample()
        return action


env = SlipperyScenario(gym.make("CartPole-v0"), epsilon=0.1)
initial_observation = env.reset()
reward_sum = 0.0
total_steps = 0

while True:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    reward_sum += reward
    total_steps += 1

    if done:
        break

print(f"Episode done in {total_steps} steps, total reward is {reward_sum}.")