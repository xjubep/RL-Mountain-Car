import gym
import matplotlib.pyplot as plt
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

env = gym.make('MountainCar-v0')

NUM_TRAIN_EPISODES = 20000
NUM_TEST_EPISODES = 50
PRINT_INTERVAL = 500
GAMMA = 1
ALPHA = 0.01
EPSILON = 0.05


def featurize_state(state):
    scaled = scaler.transform([state])
    featurized = featurizer.transform(scaled)
    return featurized


def Q_value(state, action, w):
    value = state.dot(w[action])
    return value


def evaluate_policy(state, w, n=NUM_TEST_EPISODES):
    mean_returns = 0
    for _ in range(n):
        done = False
        state = env.reset()
        state = featurize_state(state)

        while not done:
            action = action_epsilon_greedy(state, w, epsilon=0)
            state, reward, done, _ = env.step(action)
            state = featurize_state(state)
            mean_returns += reward
    return mean_returns / n


def action_epsilon_greedy(state, w, epsilon=EPSILON):
    if np.random.rand() > epsilon:
        return np.argmax([Q_value(state, a, w) for a in range(n_actions)])
    return np.random.randint(n_actions)


def train():
    for e in tqdm(range(NUM_TRAIN_EPISODES), desc='SARSA', ncols=150, unit=' episode'):
        done = False
        state = env.reset()
        state = featurize_state(state)

        while not done:
            action = action_epsilon_greedy(state, w)
            next_state, reward, done, _ = env.step(action)
            next_state = featurize_state(next_state)
            next_action = action_epsilon_greedy(next_state, w)
            returns[e] += reward

            target = reward + GAMMA * Q_value(next_state, next_action, w)
            td_error = Q_value(state, action, w) - target
            dw = td_error.dot(state)
            w[action] -= ALPHA * dw
            state = next_state

        if e % PRINT_INTERVAL == 0:
            progress[e] = evaluate_policy(state, w)


n_actions = env.action_space.n  # 3 actions (left, stop, right)
w = np.zeros((n_actions, 400))
returns = np.zeros(NUM_TRAIN_EPISODES)
progress = {}

observation_examples = np.array([env.observation_space.sample() for _ in range(10000)])
scaler = StandardScaler()
scaler.fit(observation_examples)

# Used to convert a state to a featurizes represenation.
# We use RBF kernels with different variances to cover different parts of the space
featurizer = FeatureUnion([
    ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
    ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
    ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
    ("rbf4", RBFSampler(gamma=0.5, n_components=100))
])

featurizer.fit(scaler.transform(observation_examples))

train()
plt.plot(progress.keys(), progress.values(), 'k--')
plt.title('SARSA Mountain Car')
plt.xlabel('Episode Iteration')
plt.ylabel('Return')
plt.show()
# print(progress)
