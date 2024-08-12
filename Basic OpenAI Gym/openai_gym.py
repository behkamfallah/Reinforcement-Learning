import gym

# Agent will get reward of one until it hits ground.
env = gym.make('CartPole-v0')

# We always need to reset the newly created env. Four floating-point numbers containing information about the
# x coordinate of the stick's center of mass, its speed, its angle to the platform, and its angular speed.
initial_observation = env.reset()
print(initial_observation)

# Get a sensation of action space:
print(env.action_space)  # 0: Left and 1: Right
print(env.action_space.sample())  # Returns a random sample

# Get a sensation of observation space:
print(env.observation_space)  # Range is from -inf to +inf
print(env.observation_space.sample())  # Returns a random sample

# Do a step:
# We will get a new obs, a reward, a flag and extra info which is empty
print(env.step(0))

