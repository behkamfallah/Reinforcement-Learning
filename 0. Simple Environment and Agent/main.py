from environment import Environment
from agent import Agent

env = Environment()
agent = Agent()

while not env.is_done():
    agent.step(env)

print(agent.sum_reward)
