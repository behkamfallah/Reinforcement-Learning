import gym

env = gym.make('CartPole-v0')
initial_observation = env.reset()
reward_sum = 0.0  # Because step() returns reward as a float type
total_steps = 0

while True:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    reward_sum += reward
    total_steps += 1

    if done:
        break

print(f"Episode done in {total_steps} steps, total reward is {reward_sum}.")