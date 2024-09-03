import gym
import numpy as np
import math

# A simple 2 * 3 Environment
env = gym.make("FrozenLake-v1", desc=["SFH", "HSG"], render_mode='human', is_slippery=False)
env.reset()

# Define hyperparameters here:
theta = 0.0005
discount_factor = 0.9
max_iterations = 100

# Initialize v(s) for all states here:
states = [i for i in range(env.observation_space.n)]
# Put 0 for all states
state_value = {s: 0 for s in range(env.observation_space.n)}

# Prevent confusion because states are also encoded with similar numbers.
act = {
    'LEFT': 0,
    'DOWN': 1,
    'RIGHT': 2,
    'UP': 3
}


def get_action(val):
    for key, value in act.items():
        if val == value:
            return key
    return "key doesn't exist"


# I had to manually write these because I could not find a way to
# get the next state of a transition without doing step function.
# This is useful when we are trying to compute bellman update on each state
# and we need to compute each action's value and for that we need to know what is the
# next state.
transitions = {
    (0, act['RIGHT']): 1,
    (0, act['DOWN']): 3,
    (1, act['LEFT']): 0,
    (1, act['DOWN']): 4,
    (1, act['RIGHT']): 2,
    (4, act['LEFT']): 3,
    (4, act['UP']): 1,
    (4, act['RIGHT']): 5
}

# States 2 and 3 are holes.
# State 5 is Goal.
non_terminal_states = [0, 1, 4]


# Define a Random policy
def random_policy(states):
    policy = {}
    for s in non_terminal_states:
        policy[s] = np.random.dirichlet(np.ones(env.action_space.n),size=1).tolist()
    print(f"Policy: {policy}")
    return policy


# Default Policy
policy = random_policy(states)


# Define Policy Evaluation
def iterative_policy_evaluation(policy, states, state_value, max_iterations, env):
    print('*-*-*-*-*-Iterative Policy Evaluation-*-*-*-*-*')
    k = 0
    while True:
        k += 1
        print(f"k: {k}")
        delta = 0
        for current_state in non_terminal_states:
            current_state_value = state_value[current_state]
            print(f"Current State: {current_state} and state value is : {current_state_value}")
            # Update the state value:
            state_value[current_state] = bellman_eq(env, policy, state_value, current_state, discount_factor)
            print(f"State Value: {state_value}")
            max_delta = max(delta, abs(current_state_value - state_value[current_state]))
        if max_delta < theta:
            print(f"Converged in {k} iterations.")
            break
        elif k == max_iterations:
            print(f"Terminating after {k} iterations.")
            break
    return state_value


def bellman_eq(env, policy, state_value, current_state, discount_factor):
    print("--Bellman Eq--")
    expected_value = 0
    for action in range(env.action_space.n):
        try:
            next_state = transitions[(current_state, action)]
        except:
            next_state = current_state
        print(f"Action is: {get_action(action)} and next state is {next_state}")

        if next_state == 5:
            reward = 1
        elif next_state == 2 or next_state == 3:
            reward = -1
        else:
            reward = 0
        print(f"Reward: {reward}")
        expected_value += policy[current_state][0][action] * (reward + discount_factor * state_value[next_state])
    print(f"State Value for state: {current_state} is: {expected_value}")
    return expected_value


def best_action_selector(current_state, state_value):
    print("------Best Action Selector------")
    best_action_value = -math.inf
    best_action = None

    # For each action, we should compute action-value
    for action in range(env.action_space.n):
        try:
            next_state = transitions[(current_state, action)]
        except:
            next_state = current_state
        print(f"Action is: {get_action(action)} and next state is {next_state}")

        # Reward of each state.
        if next_state == 5:
            reward = 1
        elif next_state == 2 or next_state == 3:
            reward = -1
        else:
            reward = 0

        action_value = policy[current_state][0][action] * (reward + discount_factor * state_value[next_state])
        print(f"Action Value of: {get_action(action)} is {action_value}")
        if action_value > best_action_value:
            best_action = action
            best_action_value = action_value
    print(f"Best Action is {get_action(best_action)} with value: {best_action_value}")
    return best_action, best_action_value


def policy_improvement(state_value):
    print('**********POLICY IMPROVEMENT**********')
    policy_stable = True
    for current_state in non_terminal_states:
        old_action = policy[current_state]
        print(f"Current Action is {old_action} and State is {current_state}.")
        best_action, state_value[current_state] = best_action_selector(current_state, state_value)
        # Change the probability of the best action to 1 and others to 0.
        for i in range(env.action_space.n):
            if i == best_action:
                policy[current_state][0][best_action] = 1
            else:
                policy[current_state][0][i] = 0
        print(f"Improved Policy is: {policy}")
        if old_action != policy[current_state]:
            policy_stable = False
    if policy_stable:
        return True, policy, state_value
    else:
        return False, policy, state_value


while True:
    # First Policy Evaluation is done
    state_value = iterative_policy_evaluation(policy, states, state_value, max_iterations, env)
    # Then Policy Improvement
    halt_code, new_policy, new_state_values = policy_improvement(state_value)
    # Halt code determines if Policy is Stable or not.
    if halt_code:
        break
    print(f"New Policy is: {new_policy}")
    print(f"New state values are: {new_state_values}")
