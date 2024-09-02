import gym
import numpy as np

# A simple 2 * 2 Environment
env = gym.make("FrozenLake-v1", desc=["SF", "HG"], render_mode='human', is_slippery=False)
env.reset()

# Define hyperparameters here:
delta = 0
theta = 0.01
discount_factor = 0.9
max_iterations = 10

# Initialize v(s) for all states here:
states = [i for i in range(env.observation_space.n)]
transitions = {
    (0, 1): 2,
    (0, 2): 1,
    (1, 0): 0,
    (1, 1): 3,
    (2, 3): 0,
    (2, 2): 3,
}


# Define a Random policy
def base_policy(states):
    policy = {}
    for s in states:
        policy[s] = np.random.dirichlet(np.ones(env.action_space.n),size=1).tolist()
        policy[s] = policy[s][0]
    return policy


# Default Policy
policy = base_policy(states)

# Deterministic Always win policy
#policy = {0: [0, 0, 1, 0], 1: [0, 1, 0, 0]}

print(f"Policy: {policy}")


def bellman_eq(env, policy, state_value, current_state, discount_factor):
    print("--Bellman Eq--")
    expected_value = 0
    for a in range(env.action_space.n):

        try:
            next_state = transitions[(current_state, a)]
        except:
            next_state = current_state
        print(f"Action is: {a} and next state is {next_state}")

        if next_state == 3:
            reward = 1
        elif next_state == 2:
            reward = -1
        else:
            reward = 0

        expected_value += policy[current_state][a] * (reward + discount_factor * state_value[next_state])
        print(f"Expected Value is: {expected_value}")
    return expected_value


# Define Policy Evaluation
def iterative_policy_evaluation(policy, states, max_iterations, env):
    k = 0
    state_value = {s: 0 for s in range(env.observation_space.n)}
    while True:
        k += 1
        print(f"k: {k}")
        delta = 0
        for current_state in states:
            if current_state == 0 or current_state == 1:
                print(f"Current State: {current_state}")
                current_state_value = state_value[current_state]
                # Update the state value:
                state_value[current_state] = bellman_eq(env, policy, state_value, current_state, discount_factor)
                print(f"State Value: {state_value}")
                max_delta = max(delta, abs(current_state_value - state_value[current_state]))
        if delta < theta:
            print(f"Converged in {k} iterations.")
            break
        elif k == max_iterations:
            print(f"Terminating after {k} iterations.")
            break
    return state_value


state_values = iterative_policy_evaluation(policy, states, max_iterations=10, env=env)
print(state_values)