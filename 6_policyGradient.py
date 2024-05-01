# for rl environments
import gymnasium as gym

# imports for the neural network
import torch
import torch.nn as nn
import torch.optim as optim

# other imports
import numpy as np
from collections import OrderedDict
from itertools import count
import matplotlib.pyplot as plt


# neural net
def make_torch_net(input_length: int, width: int, output_length: int, hidden=0, softmax_output=False):
    """
    Make a multilayer neural network that
    takes a vector (actually a torch tensor) of length 'input_length',
    has 'hidden' hidden layers with ReLU-activation,
    and returns a vector (actually a torch tensor) of length 'output_length'.
    If 'softmax_output' = True, then an additional softmax output layer is used.
    """
    layers = []
    layer_num = 0
    layers.append((str(layer_num), nn.Linear(input_length, width)))
    layer_num += 1
    layers.append((str(layer_num), nn.ReLU()))
    layer_num += 1
    for i in range(hidden):
        layers.append((str(layer_num), nn.Linear(width, width)))
        layer_num += 1
        layers.append((str(layer_num), nn.ReLU()))
        layer_num += 1
    layers.append((str(layer_num), nn.Linear(width, output_length)))
    if softmax_output:
        layer_num += 1
        layers.append((str(layer_num), nn.Softmax(dim=0)))
    net = nn.Sequential(OrderedDict(layers))
    print(net)
    return net


# MAIN WRAPPER CLASS
class REINFORCE(object):
    """
    REINFORCE algorithm wrapper.

    Attributes
    ----------
    env : gymnasium.env
        Defaults to the "CartPole-v1" environment.
    device : cuda.device
        The hardware used by torch in computation
    memory : memory
        The transition memory.
    n_actions : int
        Number of actions in the environment.
    n_observations : int
        Dimensionality of the state vector of the environment.
    episode_durations : list[int]
        A list of the lengths of past episodes.
    """

    def __init__(self, env=None, memory_length=1000):
        self.env = gym.make("CartPole-v1") if env is None else env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # number of actions in gym environment
        self.n_actions = self.env.action_space.n
        # dimensionality of state observations in gym environment
        state, _ = self.env.reset()
        self.n_observations = len(state)
        self.episode_durations = []

    def initialize_action_net(self, width=128, hidden=1):
        """
        Set up an action net.
        """
        self.action_net = make_torch_net(input_length=self.n_observations, width=width, output_length=self.n_actions,
                                         hidden=hidden, softmax_output=True)

    def _use_action_net(self, state):
        """
        Use the policy net to to get an action for a state.
        """
        act_prob = self.action_net(torch.from_numpy(state).float())
        return np.random.choice(np.array([0, 1]), p=act_prob.data.numpy())

    def plot_durations(self, averaging_window=50, title=""):
        """
        Visually represent the learning history to standard output.
        """
        averages = []
        for i in range(1, len(self.episode_durations) + 1):
            lower = max(0, i - averaging_window)
            averages.append(sum(self.episode_durations[lower:i]) / (i - lower))
        plt.xlabel("Episode")
        plt.ylabel("Episode length with " + str(averaging_window) + "-running average")
        plt.title(title)
        plt.plot(averages, color="black")
        plt.scatter(range(len(self.episode_durations)), self.episode_durations, s=2)
        plt.show()

    def play(self, num_steps=500, env_name="CartPole-v1"):
        """
        Play 'num_steps' using the current policy network.
        During play the environment is rendered graphically.
        """
        # initialize the environment and get the state
        env = gym.make(env_name, render_mode="human")
        state, _ = env.reset()
        for i in range(num_steps):
            action = self._use_action_net(state)
            state, _, terminated, truncated, _ = env.step(action.item())
            if terminated or truncated:
                print("Number of steps: " + str(i) + ".")
                break
        env.close()

    def learn(self, num_episodes=500, gamma=0.99, learning_rate=3e-3):
        """
        Train using the REINFORCE algorithm.

        Parameters
        ----------
        num_episodes : int
            Number of rounds to be played.
            Note that a round lasts longer if the model is good.
            Hence the time a round takes increases during training.
        batch_size : int
            Size of the dataset sampled from the memory for a gradient step.
        gamma : float
            Discount rate.
        learning_rate : float
            Learning rate for the adam optimizer of torch.
        """
        # set up torch optimizer
        optimizer = optim.Adam(self.action_net.parameters(), lr=learning_rate, amsgrad=False)

        # Loops over episodes
        for episode in range(num_episodes):
            episode_rewards = []
            episode_log_probs = []

            # Generates an episode
            state, _ = self.env.reset()
            for t in count():
                # Selects action using the policy network
                action = self._use_action_net(state)
                log_prob = torch.log(self.action_net(torch.from_numpy(state).float()))[action]

                # Takes action in the environment
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode_rewards.append(reward)
                episode_log_probs.append(log_prob)

                if terminated or truncated:
                    break
                state = next_state

            # Calculates returns
            returns = np.zeros_like(episode_rewards, dtype=np.float32)
            G = 0
            for t in reversed(range(len(episode_rewards))):
                G = gamma * G + episode_rewards[t]
                returns[t] = G

            # Calculates policy gradients and update weights
            policy_loss = []
            for log_prob, G in zip(episode_log_probs, returns):
                policy_loss.append(-log_prob * G)
            policy_loss = torch.stack(policy_loss).sum()

            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

            # Tracks episode duration
            self.episode_durations.append(len(episode_rewards))

            # Prints episode info
            if episode % 10 == 0:
                print("Episode {}\tLast Episode Length: {}".format(episode, len(episode_rewards)))


# EXAMPLE
mylearner = REINFORCE()
mylearner.initialize_action_net(width=256, hidden=0)
mylearner.learn()
mylearner.plot_durations()
