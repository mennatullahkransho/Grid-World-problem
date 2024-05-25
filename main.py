import numpy as np

# Parameters
num_states = 12
num_actions = 4
actions = [(1, 0), (0, -1), (-1, 0), (0, 1)]  # Down, Left, Up, Right
gamma = 0.9  # Discount factor
noise = 0.2  # Probability of veering off
num_iterations = 100

# Rewards and grid setup
rewards = np.full((3, 4), -0.01)  # Default rewards
rewards[0, 3] = 1.0  # Reward for state 3
rewards[1, 3] = -1.0  # Reward for state 7

# Walls and terminal states
walls = [(1, 1)]  # Wall at state 5
terminals = [(0, 3), (1, 3)]  # Terminal states at 3 and 7

# Initialize value grid
values = np.zeros((3, 4))

def transition(state, action):
    """Returns the next state and reward given an action."""
    if state in terminals:  # No action if state is terminal
        return state, 0
    
    next_state = (state[0] + action[0], state[1] + action[1])
    
    # Check for walls and grid edges
    if next_state in walls or not (0 <= next_state[0] < 3 and 0 <= next_state[1] < 4):
        next_state = state  # Remain in current state if move is invalid
    
    return next_state, rewards[next_state]

def value_iteration():
    for _ in range(num_iterations):
        new_values = np.copy(values)
        for i in range(3):
            for j in range(4):
                state = (i, j)
                if state in walls or state in terminals:
                    continue
                v = []
                for action in actions:
                    next_state, reward = transition(state, action)
                    # Compute the expected value
                    v.append(reward + gamma * values[next_state])
                new_values[i, j] = max(v)  # Update with the maximum value
        if np.allclose(values, new_values, atol=1e-4):
            break
        values[:] = new_values

def extract_policy():
    policy = np.full((3, 4), ' ', dtype='<U5')  # Initialize policy grid
    for i in range(3):
        for j in range(4):
            state = (i, j)
            if state in terminals:
                policy[i, j] = '+' if rewards[state] > 0 else '-'
                continue
            if state in walls:
                policy[i, j] = 'WALL'
                continue
            best_action_value = float('-inf')
            best_action = None
            for k, action in enumerate(actions):
                next_state, reward = transition(state, action)
                action_value = reward + gamma * values[next_state]
                if action_value > best_action_value:
                    best_action_value = action_value
                    best_action = k
            policy[i, j] = ['Down', 'Left', 'Up', 'Right'][best_action]
    return policy

value_iteration()
print(values)

policy = extract_policy()
print(policy)


