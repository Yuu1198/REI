# for rl environments
import gymnasium as gym

# imports for the neural network
import torch
import torch.nn as nn
import torch.optim as optim

# other imports
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque, OrderedDict
from itertools import count

# convenience class to keep transition data straight
# is used inside 'replayMemory'
transitionData = namedtuple('Transition', ['state', 'action', 'next_state', 'reward'])


class replayMemory(object):
    """
    Store transitions consisting of 'state', 'action', 'next_state', 'reward'.

    Attributes
    ----------
    memory : collections.deque
        Here we store named tuples of the class 'transitionData'.
        A deque is a data structure that allows appending/popping "from the right" and "from the left":
        https://en.wikipedia.org/wiki/Double-ended_queue
    """

    def __init__(self, length: int):
        # the deque class is designed for popping from the right and from the left
        self.memory = deque([], maxlen=length)

    def save(self, state, action, next_state, reward):
        """
        Save the transition consisting of 'state', 'action', 'next_state', 'reward'.
        """
        self.memory.append(transitionData(state, action, next_state, reward))

    def sample(self, batch_size: int):
        """
        Bootstrap 'batch_size' transitions from the memory.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        Get the number of transitions in memory.
        """
        return len(self.memory)


# Q-NETWORK
def make_torch_net(input_length: int, width: int, output_length: int, hidden=0):
    """
    Make a multilayer perceptron that
    takes a vector (actually a torch tensor) of length 'input_length',
    has 'hidden' hidden layers with ReLU-activation,
    and returns a vector (actually a torch tensor) of length 'output_length'.
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
    net = nn.Sequential(OrderedDict(layers))
    print(net)
    return net


# MAIN WRAPPER CLASS
class deepQ(object):
    """
    Deep Q Learning wrapper.

    Attributes
    ----------
    env : gymnasium.env
        Defaults to the "CartPole-v1" environment.
    device : cuda.device
        The hardware used by torch in computation
    memory : replayMemory
        The transition memory of the q-learner.
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
        self.memory = replayMemory(length=memory_length)
        # number of actions in gym environment
        self.n_actions = self.env.action_space.n
        # dimensionality of state observations in gym environment
        state, _ = self.env.reset()
        self.n_observations = len(state)
        self.episode_durations = []

    def initialize_networks(self, width=128, hidden=1):
        """
        Sets up policy and target networks with 'hidden' hidden layers of given 'width'.
        Activations are ReLU.
        The dimensionalities of input and output layer are chosen automatically.
        """
        # set up policy net
        self.policy_net = make_torch_net(input_length=self.n_observations, width=width, output_length=self.n_actions,
                                         hidden=hidden).to(self.device)
        # set up target net
        self.target_net = make_torch_net(input_length=self.n_observations, width=width, output_length=self.n_actions,
                                         hidden=hidden).to(self.device)
        # copy parameters of policy net to target net
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _eps_greedy_action(self, state, eps):
        """
        Returns an 'eps'-greedy action.
        Does not modify the object.
        """
        if random.random() > eps:
            # deactivating grad computation in torch makes it a little faster
            with torch.no_grad():
                # t.max(1) returns the largest column value of each row
                # the second column of the result is the index of the maximal element
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

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
        # coerce the state to torch tensor type
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        for i in range(num_steps):
            action = self._eps_greedy_action(state, eps=0)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            if done:
                break
            else:
                state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        env.close()

    def learn(self, num_episodes=500, batch_size=100, gamma=0.99, eps_start=0.9, eps_end=0.01, eps_decay=1000,
              target_net_update_rate=0.005, learning_rate=1e-4):
        """
        Train using the deep q-learning algorithm.

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
        eps_start : float
            Epsilon for epsilon-greedy action selection at the start of training.
        eps_end : float
            Epsilon for epsilon-greedy action selection at the end of training.
        eps_decay : float
            Decay rate of epsilon for epsilon-greedy action selection during training.
        target_net_update_rate : float
            Should be between 0 and 1.
            Determines the mix-in rate of the parameters of the policy network into the target network.
            This ensures that the target network lags behind enough to stabilize the learning task.
        learning_rate : float
            Learning rate for the adam optimizer of torch.
        """
        # Init Adam optimizer
        optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        # MSE loss function
        criterion = nn.MSELoss()

        # Training loop
        for episode in range(num_episodes):
            # Reset for consistent learning process and exploration
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            episode_reward = 0

            # Until episode termination
            for t in count():
                # Calculates epsilon (decays over episodes --> agent exploits more)
                epsilon = eps_end + (eps_start - eps_end) * math.exp(-1.0 * episode / eps_decay)
                # Selects action based on epsilon greedy policy
                action = self._eps_greedy_action(state, epsilon)
                # Takes action obtains next state, reward and if done
                step_result = self.env.step(action.item())
                next_state, reward, done, _ = step_result[:4]
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
                # Updates reward
                episode_reward += reward

                # Checks if episode terminates
                if done:
                    next_state = None

                # Saves transition to replay memory
                self.memory.save(state, action, next_state, torch.tensor([reward], dtype=torch.float32))

                # Updates state
                state = next_state

                # Gradient Descent every batch size steps
                if len(self.memory) >= batch_size:
                    # Samples a batch of transitions
                    transitions = self.memory.sample(batch_size)
                    batch = transitionData(*zip(*transitions))

                    # Filters transitions which have no next state (= end of episode)
                    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                                  device=self.device, dtype=torch.bool)
                    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

                    # Separates states, actions, and rewards of the batch
                    state_batch = torch.cat(batch.state)
                    action_batch = torch.cat(batch.action)
                    reward_batch = torch.cat(batch.reward)

                    # Calculates predicted q-values for each state and action in the batch
                    state_action_values = self.policy_net(state_batch).gather(1, action_batch)

                    next_state_values = torch.zeros(batch_size, device=self.device)
                    next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

                    # Calculates target q-values based on the Bellman equation
                    expected_state_action_values = (next_state_values * gamma) + reward_batch

                    # Calculates the loss between the predicted q-values and the target q-values using MSE
                    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Finish if episode has ended
                if done:
                    break

            # Updates duration of episodes
            self.episode_durations.append(t + 1)
            # Updates target network to match policy network
            if episode % target_net_update_rate == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())


# EXAMPLE
myqlearner = deepQ()
myqlearner.initialize_networks()
myqlearner.learn()

# Plot learning history
myqlearner.plot_durations(title='Learning History')

