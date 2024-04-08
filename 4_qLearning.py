import numpy as np


class treasureHunt(object):
    """
    This represents a two-dimensional gridworld labyrinth with a fixed start state and a "treasure" positions on the grid.
    If the agent finds the treasure it gets a reward and it teleported back to the start position.

    Attributes
    ----------
    matrix : list[list[int]]
        An array representing the gridworld.
        Coding:
        '0' free position
        '1' Wall
        '2' Start position
        '3' Treasure
    start : [int,int]
        Coordinates of the start position.
    treasure : [int,int]
        Coordinates of the treasure.
    position : [int,int]
        Current position of the agent.
    reward: float
        Last received reward.
    num_visits : list[list[int,int]]
        Array storing the number of times the respective coordinate has been visited by the agent.
        Array dimensions coincide with the dimensions of 'matrix'.
    q_u : list[list[float]]
        Array storing the q-values of the action 'up' for the respective position.
        Array dimensions coincide with the dimensions of 'matrix'.
    q_d : list[list[float]]
        Array storing the q-values of the action 'down' for the respective position.
        Array dimensions coincide with the dimensions of 'matrix'.
    q_r : list[list[float]]
        Array storing the q-values of the action 'right' for the respective position.
        Array dimensions coincide with the dimensions of 'matrix'.
    q_l : list[list[float]]
        Array storing the q-values of the action 'left' for the respective position.
        Array dimensions coincide with the dimensions of 'matrix'.
    reward_history : list[float]
        Stores the received rewards.
    """

    def __init__(self, matrix=None):
        import numpy as np
        # default matrix if not given
        if matrix is None:
            matrix = [[1 for i in range(10)]] + [[1] + [0 for i in range(8)] + [1] for j in range(8)] + [
                [1 for i in range(10)]]
            matrix[1][1] = 2
            matrix[1][8] = 3
            matrix = np.array(matrix, dtype=int)
        else:
            # some checks whether the input matrix is valid
            unique_entries, counts = np.unique(matrix, return_counts=True)
            assert (all([x in unique_entries for x in range(4)])), "Something seems to be wrong with the input matrix."
            index_2 = np.where(unique_entries == 2)[0][0]
            index_3 = np.where(unique_entries == 3)[0][0]
            assert (counts[index_2] == 1), "Something seems to be wrong with the start position."
            assert (counts[index_3] == 1), "Something seems to be wrong with the treasure position."
        # attributes describing the environment
        self.matrix = matrix
        start = [x.tolist()[0] for x in np.where(matrix == 2)]
        self.start = start
        self.treasure = [x.tolist()[0] for x in np.where(matrix == 3)]
        # attributes describing the state of the environment
        self.position = start
        self.reward = None
        self.num_visits = np.zeros(matrix.shape, dtype=int)
        self.q_u = np.zeros(matrix.shape)
        self.q_d = np.zeros(matrix.shape)
        self.q_r = np.zeros(matrix.shape)
        self.q_l = np.zeros(matrix.shape)
        # attributes storing a history
        self.reward_history = []

    def reset(self, reset_q_values=True):
        """
        Resets the attributes of the object. Q-values may be retained.
        """
        import numpy as np
        self.position = self.start
        self.reward = None
        self.num_visits = np.zeros(self.matrix.shape, dtype=int)
        self.reward_history = []
        if reset_q_values:
            self.q_u = np.zeros(self.matrix.shape)
            self.q_d = np.zeros(self.matrix.shape)
            self.q_r = np.zeros(self.matrix.shape)
            self.q_l = np.zeros(self.matrix.shape)

    @staticmethod
    def _update_average(old_average, num_obs, new_obs):
        new_average = old_average * num_obs / (num_obs + 1) + new_obs / (num_obs + 1)
        return new_average, num_obs + 1

    @staticmethod
    def sample_Boltzmann(q_list, temperature):
        """
        Softmax sampler. Returns an index of 'q_list'.
        """
        import numpy as np
        from numpy.random import choice

        def logsumexp(x):
            c = np.array(x).max()
            return c + np.log(np.sum(np.exp(x - c)))

        q_weighted = [x / temperature for x in q_list]
        prob_vec = list(np.exp(q_weighted - logsumexp(q_weighted)))
        return choice(range(len(q_list)), p=prob_vec)

    def _get_action_set(self):
        """
        Return the possible actions for the current position of the agent.
        Actions are coded as:
        'u' up
        'd' down
        'r' right
        'l' left
        """
        action_set = []
        if self.matrix[self.position[0] + 1][self.position[1]] != 1:
            action_set.append("d")
        if self.matrix[self.position[0]][self.position[1] + 1] != 1:
            action_set.append("r")
        if self.matrix[self.position[0] - 1][self.position[1]] != 1:
            action_set.append("u")
        if self.matrix[self.position[0]][self.position[1] - 1] != 1:
            action_set.append("l")
        return action_set

    def take_action(self, action):
        """
        Takes an ction and updates the attributes, 'position', 'reward', 'num_visits', and 'reward_history'.
        """
        if action not in self._get_action_set():
            raise ValueError("This is not a valid action.")
        # compute next coordinate
        if action == "u":
            pos = [self.position[0] - 1, self.position[1]]
        if action == "d":
            pos = [self.position[0] + 1, self.position[1]]
        if action == "r":
            pos = [self.position[0], self.position[1] + 1]
        if action == "l":
            pos = [self.position[0], self.position[1] - 1]
        # enact environment dynamics
        if pos == self.treasure:
            self.position = self.start
            self.reward = 1
        else:
            self.position = pos
            self.reward = 0
        # update state count
        self.num_visits[self.position[0]][self.position[1]] += 1
        # append to history
        self.reward_history.append(self.reward)

    def plot_reward_history(self, title=""):
        """
        Plot the reward history to standard output.
        """
        import matplotlib.pyplot as plt
        averages = [self.reward_history[0]]
        for i in range(1, len(self.reward_history)):
            averages.append(self._update_average(averages[-1], i, self.reward_history[i])[0])
        plt.xlabel("Time")
        plt.ylabel("Running average of rewards")
        plt.title(title)
        plt.plot(averages, color="black")
        plt.show()

    def print_map(self):
        """
        Print a representation of the gridworld to standard output.
        """

        def translator(x):
            if x == 1:
                return "X"
            if x == 2:
                return "S"
            if x == 3:
                return "T"
            else:
                return " "

        string = ""
        for i in range(len(self.matrix)):
            string += "\n"
            for j in range(len(self.matrix[0])):
                string += translator(self.matrix[i][j]) + " "
        print(string)

    def play_greedy(self, timesteps=100):
        """
        Play greedily with respect to the q-values for 'timesteps' time.
        """
        import numpy as np
        for t in range(timesteps):
            action_set = self._get_action_set()
            q_values = [getattr(self, "q_" + action)[self.position[0]][self.position[1]] for action in action_set]
            action = action_set[np.argmax(q_values)]
            self.take_action(action)

    def visualize_greedy_policy(self):
        """
        Print a visualization of the greedy policy to standard output.
        """
        import numpy as np
        visualization = np.full((len(self.matrix), len(self.matrix)), "X")
        for i in range(0, len(self.matrix)):
            for j in range(0, len(self.matrix)):
                if self.matrix[i, j] == 0 or self.matrix[i, j] == 2:
                    ind = np.argmax([self.q_d[i, j], self.q_u[i, j], self.q_l[i, j], self.q_r[i, j]])
                    if ind == 0:
                        visualization[i][j] = "v"
                    if ind == 1:
                        visualization[i][j] = "^"
                    if ind == 2:
                        visualization[i][j] = "<"
                    if ind == 3:
                        visualization[i][j] = ">"
                if self.matrix[i, j] == 3:
                    visualization[i][j] = "T"
        string = ""
        for row in visualization:
            string += "\n"
            for entry in row:
                string += entry + " "
        print(string)

    def q_learner(self, timesteps=1000, boltzmann_schedule=lambda t: 1, discount_factor=0.9,
                  learning_rate_schedule=lambda t: 1 / (t + 1)):
        """
        Learns Q-values for state action pairs using q-learning algorithm.
        """
        # Iterations of Q-learning
        for t in range(timesteps):
            # Gets available actions for current position/state
            action_set = self._get_action_set()
            # Stores q-values for each action in the action set
            q_values = []
            for action in action_set:
                q_value = getattr(self, f'q_{action}')[self.position[0]][self.position[1]]
                q_values.append(q_value)
            # Chooses action based on q-value and temperature
            action_index = self.sample_Boltzmann(q_values, boltzmann_schedule(t))
            action = action_set[action_index]

            # Takes the chosen action and updates states
            current_position = self.position
            self.take_action(action)
            new_position = self.position

            # Gets Q-value array of taken action
            q_array = getattr(self, f'q_{action}')
            # Gets Q-value at current position
            current_q_value = q_array[current_position[0]][current_position[1]]

            # Gets Q-value arrays of each action in action set of next state
            new_q_arrays = [getattr(self, f'q_{a}') for a in action_set]
            # Gets Q-values at the coordinates of the new state for each action
            q_values = [q_array[new_position[0]][new_position[1]] for q_array in new_q_arrays]

            # Updates the Q-value
            q_array[current_position[0]][current_position[1]] += (learning_rate_schedule(t) *
                                                                  (self.reward + discount_factor * max(q_values) -
                                                                   current_q_value))

    def sarsa_learner(self, timesteps=1000, boltzmann_schedule=lambda t: 1, discount_factor=0.9,
                      learning_rate_schedule=lambda t: 1 / (t + 1)):
        """
            Learns Q-values for state-action pairs using SARSA algorithm.
        """
        # Iterations of SARSA learning
        for t in range(timesteps):
            # Gets available actions for current position
            action_set = self._get_action_set()
            # Stores q-values for each action of current position in the action set
            q_values = []
            for action in action_set:
                q_value = getattr(self, f'q_{action}')[self.position[0]][self.position[1]]
                q_values.append(q_value)
            # Chooses action based on q-value and temperature
            action_index = self.sample_Boltzmann(q_values, boltzmann_schedule(t))
            action = action_set[action_index]
            # Takes the chosen action and updates states
            self.take_action(action)
            current_position = self.position

            # Gets available actions for current position
            action_set = self._get_action_set()
            # Chooses next action based on q-value and temperature using Boltzmann exploration
            next_q_values = []
            for next_action in action_set:
                q_values_array = getattr(self, f'q_{next_action}')
                next_q_value = q_values_array[self.position[0]][self.position[1]]
                next_q_values.append(next_q_value)
            next_action_index = self.sample_Boltzmann(next_q_values, boltzmann_schedule(t))
            next_action = action_set[next_action_index]
            # Takes the next action and updates states
            self.take_action(next_action)
            new_position = self.position

            # Gets Q-value array of taken action
            q_array = getattr(self, f'q_{action}')
            # Gets Q-value at coordinates of the old state
            current_q_value = q_array[current_position[0]][current_position[1]]
            # Gets Q-value of the next action at the coordinates of the new state
            next_q_array = getattr(self, f'q_{next_action}')
            next_q_value = next_q_array[new_position[0]][new_position[1]]

            # Updates the Q-value
            q_array[current_position[0]][current_position[1]] += (learning_rate_schedule(t) *
                                                          (self.reward + discount_factor * next_q_value -
                                                           current_q_value))


# Defines environment
mymap = np.zeros((9, 9))
mymap[0,] = 1
mymap[8,] = 1
mymap[:, 0] = 1
mymap[:, 8] = 1
mymap[1, 4] = 1
mymap[7, 4] = 1
mymap[3:6, 4] = 1
mymap[1, 1] = 2
mymap[1, 7] = 3

# Creates environment
example = treasureHunt(matrix=mymap)
print("Map:")
example.print_map()

# Reset
example.reset()

# Q-Learning
example.q_learner(timesteps=10000)
print("Q-learning:")
example.visualize_greedy_policy()
example.plot_reward_history()

# Reset
example.reset()

# SARSA
example.sarsa_learner(timesteps=10000)
print("SARSA:")
example.visualize_greedy_policy()
example.plot_reward_history()
