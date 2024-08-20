import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple


# Define Neural Network
class NNRegression(nn.Module):
    def __init__(self, ob_size, hid_size, n_actions):
        super(NNRegression, self).__init__()
        self.layer1 = nn.Linear(ob_size, hid_size)
        self.layer2 = nn.Linear(hid_size, n_actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


# Create Environment
env = gym.make("CartPole-v1", render_mode='human')
# Get initial Observation
initial_observation = env.reset()
print("Initial Observation: ", initial_observation)

# Get a sensation of action space:
num_actions = env.action_space.n  # '.n' makes it '2' instead of Discrete(2)
print("Action Space: ", env.action_space)  # 0: Left and 1: Right
print("Action Space Sample: ", env.action_space.sample())  # Returns a random sample

# Get a sensation of observation space:
obs_size = env.observation_space.shape[0]  # Outputs '4', '.shape' would have given '(4,)'
print("Observation Space: ", env.observation_space)  # Range is from -inf to +inf
print("Observation Space Sample: ", env.observation_space.sample())  # Returns a random sample
print('--------------------------------------------------------------------------------------------')
# Show Environment
# env.render()

# Create an instance of Neural Network
hidden_size = 128
agent = NNRegression(obs_size, hidden_size, num_actions)

# Initialize Loss function and Optimizer
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=agent.parameters(), lr=0.03)

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


# PLAY N EPISODE
def iterate_batches(environment, net, batch_size):
    # batch is a list. Each element will be about one episode
    # and will contain total reward, observations, and actions in that episode.
    batch = []

    # Each element in this list is a tuple named 'EpisodeStep' which has two elements named: observation and action.
    episode_steps = []

    episode_reward = 0.0
    current_observation = environment.reset()
    current_observation = current_observation[0]
    # The softmax function is often used as the last activation function of
    # a neural network to normalize the output of a network to a probability
    # distribution over predicted output classes.
    sm = nn.Softmax(dim=1)
    while True:
        # Turn observation into tensor in order to be able to feed it to NN.
        obs_tensor = torch.FloatTensor([current_observation])

        # First feed the observation to NN then turn the output to a prob dist and save it as tensor.
        action_prob_tensor = sm(net(obs_tensor))
        action_probs = action_prob_tensor.data.numpy()[0]

        # 'p=action_probs' is the probabilities associated with each entry in 'len(action_probs)'.
        action = np.random.choice(len(action_probs), p=action_probs)

        # Now, step:
        next_obs, reward, done, truncated, info = environment.step(action)
        episode_reward += reward

        # Save the observation and action
        step = EpisodeStep(observation=current_observation, action=action)
        episode_steps.append(step)

        if done:
            # Save finished episode in batch.
            finished_episode = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(finished_episode)

            # Reset:
            episode_steps = []
            episode_reward = 0.0
            next_obs = env.reset()
            next_obs = next_obs[0]

            if len(batch) == batch_size:
                yield batch
                batch = []

        current_observation = next_obs

# DEFINE DECISION BOUNDARY AND THROW AWAY
def filter_best_batches(batch, percentile):
    rewards = [s.reward for s in batch]
    reward_boundary = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    observations_train = []
    actions_train = []

    for reward, steps in batch:
        if reward > reward_boundary:
            observations_train.extend([step.observation for step in steps])
            actions_train.extend([step.action for step in steps])

    # Turn them into tensor and return
    train_obs_v = torch.FloatTensor(observations_train)
    train_act_v = torch.LongTensor(actions_train)
    return train_obs_v, train_act_v, reward_boundary, reward_mean


BATCH_SIZE = 16
PERCENTILE = 80
for iter_no, batch in enumerate(iterate_batches(env, agent, BATCH_SIZE)):
    obs_tensor, acts_tensor, reward_bound, reward_avg = filter_best_batches(batch, PERCENTILE)
    optimizer.zero_grad()
    action_scores_v = agent(obs_tensor)
    loss_v = loss(action_scores_v, acts_tensor)
    loss_v.backward()
    optimizer.step()
    print(f"Iteration No.:{iter_no}, Loss: {loss_v.item()}, Reward Average: {reward_avg}, Reward Bound: {reward_bound}")
    if reward_avg > 199:
        print("Solved!")
        break
