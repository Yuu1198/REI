class MDP(object):
    """
    This represents a Markov decision process. The representable MDPs are restricted to those that have the same action set at every state.

    Attributes
    ----------
    states : list[object]
        A list of the states of the MDP.
    actions : list[object]
        A list of the actions of the MDP.
    transition_map : function
        This (possibly propabilistic) map takes a tuple (state, action) to a tuple (next_state, reward) where 'next_state' is from 'states'.
        This map must be defined on all possible combinations (A,B) where A is from 'states' and B is from 'actions'.
    state : object
        The current state of the MDP.
    reward : float
        The last reward of the MDP.
    reward_history : list[float]
        A list of the rewards that have been observed in the past.
    action_history : list[object]
        A list of the historical actions.
    state_history : list[object]
        A list of the historical states.
    q_values : list[list[float]]
        An array of state-action values. First coordinate: state. Second coordinate: action.
    """

    def __init__(self, states, actions, transition_map=None):
        # attributes describing the mdp
        self.states = states
        self.actions = actions
        self.transition_map = transition_map
        # attributes describing the current state and policy
        self.q_values = [[0] * len(actions) for i in range(len(states))]
        self.state = None
        self.reward = None
        # attributes describing the history
        self.reward_history = []
        self.action_history = []
        self.state_history = []

    @staticmethod
    def _update_average(old_average: float, num_obs: int, new_obs: float):
        """
        Helper function for online averaging.
        """
        # online averaging formula
        return old_average * num_obs / (num_obs + 1) + new_obs / (num_obs + 1)

    @staticmethod
    def _get_index(entry: float, vector):
        """
        Helper function to get the first index of a vector containing a given entry.
        """
        for i in range(len(vector)):
            if vector[i] == entry:
                return i

    def reset(self, reset_q_values: bool = True):
        """
        Resets all history attributes, and initializes the state.
        """
        from random import choice
        self.state = choice(self.states)
        self.reward_history = []
        self.action_history = []
        self.state_history = []
        if reset_q_values:
            self.q_values = [[0] * len(self.actions) for i in range(len(self.states))]

    def take_action(self, action):
        """
        Takes the action, adjusts the objects state, reward, and history attributes accordingly.
        'Action' must be in 'actions'.
        """
        next_state, reward = self.transition_map(self.state, action)
        self.reward = reward
        self.state = next_state
        # append to histories
        self.reward_history.append(self.reward)
        self.action_history.append(action)
        self.state_history.append(self.state)

    def eps_greedy_action(self, epsilon: float):
        """
        Return an action according to an epsilon-greedy policy with respect to q-values.
        Does not modify the object.
        """
        if epsilon < 0 or epsilon > 1:
            raise ValueError("Epsilon is not in the interval [0,1].")
        from numpy.random import binomial, choice
        if binomial(1, epsilon) == 1:
            index = choice(range(len(self.actions)))
            return self.actions[index]
        else:
            from numpy import argmax
            for i in range(len(self.states)):
                if self.states[i] == self.state:
                    state_index = i
                    break
            return self.actions[argmax(self.q_values[state_index])]

    def softmax_action(self, beta: float):
        """
        Return an action according to a softmax policy with respect to q-values.
        Does not modify the object.
        """
        import numpy as np
        current_state_index = self.states.index(self.state)
        current_state_q_values = np.array(self.q_values[current_state_index])

        # Assign probability to each action based on the q-value
        action_probabilities = np.exp(current_state_q_values / beta)
        # Normalizes probabilities --> probability distribution
        action_probabilities /= np.sum(action_probabilities)

        # Takes action based on probability
        return np.random.choice(self.actions, p=action_probabilities)

    def sarsa(self, timesteps: int, beta_schedule: callable, discount_factor: float, learning_rate_schedule: callable):
        """
        The SARSA algorithm.
        """
        for _ in range(timesteps):
            # Choose action
            selected_action = self.softmax_action(beta_schedule())
            current_state_index = self.states.index(self.state)
            current_action_index = self.actions.index(selected_action)

            # Take action, obtain reward, observe next state
            self.take_action(selected_action)

            # Simulate next action
            next_action = self.softmax_action(beta_schedule())
            next_state_index = self.states.index(self.state)
            next_action_index = self.actions.index(next_action)

            # Update
            self.q_values[current_state_index][current_action_index] += (
                    learning_rate_schedule() *
                    (self.reward + discount_factor * self.q_values[next_state_index][next_action_index] -
                     self.q_values[current_state_index][current_action_index]))

    def plot_reward_history(self, title: str = ""):
        """
        Displays a graphical representation of the reward-history to standard output.
        The average reward of the past is displayed.
        """
        import matplotlib.pyplot as plt
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Reward")
        running_average = [self.reward_history[0]]
        for i in range(1, len(self.reward_history)):
            running_average.append(self._update_average(running_average[i - 1], i + 1, self.reward_history[i]))
        plt.plot(running_average, color="black")
        plt.show()

    def plot_action_history(self, title: str = ""):
        """
        Displays a graphical representation of the action-history to standard output.
        The fraction of times an action has been chosen in the past is displayed.
        """
        import matplotlib.pyplot as plt
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Fraction of action in past")
        for i in set(self.action_history):
            actions_chosen = [1 if x == i else 0 for x in self.action_history]
            actions_percent = [actions_chosen[0]]
            for t in range(1, len(actions_chosen)):
                actions_percent.append(self._update_average(actions_percent[t - 1], t + 1, actions_chosen[t]))
            plt.plot(actions_percent)
        plt.show()

    def plot_state_history(self, title: str = ""):
        """
        Displays a graphical representation of the state-history to standard output.
        The fraction of times a state has been visited in the past is displayed.
        """
        import matplotlib.pyplot as plt
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Fraction of state in past")
        for i in set(self.state_history):
            states_visited = [1 if x == i else 0 for x in self.state_history]
            states_percent = [states_visited[0]]
            for t in range(1, len(states_visited)):
                states_percent.append(self._update_average(states_percent[t - 1], t + 1, states_visited[t]))
            plt.plot(states_percent)
        plt.show()

    def print_q_values(self, title: str = "Q-Values:"):
        """
        Prints the q-values to standard output.
        """
        string = title + "\n"
        for state in self.states:
            for action in self.actions:
                string += "(State: " + str(state) + ", Action: " + str(action) + "): "
                string += str(
                    self.q_values[self._get_index(state, self.states)][self._get_index(action, self.actions)]) + ".\n"
        print(string)


# Solution to last week's exercise. To be used as the 'transition_map' attribute of the 'MDP' class.
def opponent(state, action):
    """
    Returns tuple (next_state, reward).
    Corresponds to the repetition-avoiding rock-paper-scissors player described in the last exercise sheet.
    Actions and states are coded as strings for clarity. Rock: "r". Paper: "p". Scissors: "s".
    """
    from random import choice
    states = ["r", "p", "s"]
    states.remove(state)
    newstate = choice(states)
    if action == "r":
        if newstate == "r":
            return newstate, 0
        if newstate == "p":
            return newstate, -1
        if newstate == "s":
            return newstate, 1
    if action == "p":
        if newstate == "r":
            return newstate, 1
        if newstate == "p":
            return newstate, 0
        if newstate == "s":
            return newstate, -1
    if action == "s":
        if newstate == "r":
            return newstate, -1
        if newstate == "p":
            return newstate, 1
        if newstate == "s":
            return newstate, 0


# Definitions
states = ['r', 'p', 's']
actions = ['r', 'p', 's']

# MDP
mdp = MDP(states=states, actions=actions, transition_map=opponent)
mdp.reset()

# SARSA
timesteps = 10000
epsilon = 0.1
discount_factor = 0.9
learning_rate = 0.1
beta = lambda: 0.1
mdp.sarsa(timesteps, beta, discount_factor, lambda: learning_rate)

# Plot
mdp.plot_reward_history(title="Reward History")
mdp.plot_action_history(title="Action History")
mdp.plot_state_history(title="State History")
mdp.print_q_values()
