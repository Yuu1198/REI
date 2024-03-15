# This script requires
# matplotlib >= 3.5.1
# numpy >= 1.22.2
# math


"""
The aim of this script is to simulate a situation in which an agent repeatedly has to choose
between the same options, of which there are finitely many.
Each option leads to a reward drawn from a fixed Gaussian distribution: a multi-armed bandit with Gaussian rewards.
The aim of the agent is to repeatedly interact with the environment to learn which arm to pull.
To do so the agent maintains numerical estimates for the value of each arm (action-value method).

In this exercise some methods have been removed, you can recognize them by the 'pass' statement.
"""
import math
import random


class GaussianBandit(object):
    """
    This represents a multi-armed bandit whose arms have normally distributed rewards.

    Attributes
    ----------
    means_list : list[float]
        An ordered list of the expected values of the arms.
    stdev_list : list[float]
        An ordered list of the standard deviations of the arms.
    q_values : list[float]
        An ordered list of q-values for the arms.
    action_count : list[int]
        An ordered list whose entries are the number of times an arm has been pulled.
    action_history : list[int]
        A list of the actions that have been taken in the past.
    reward_history : list[float]
        A list of the rewards that have been observed in the past.
    q_history : list[list[float]]
        A list of the historical q-value vectors.
    """

    def __init__(self, means_list, stdev_list=None):
        if stdev_list != None and len(means_list) != len(stdev_list):
            raise ValueError('The list of means and the list of standard deviations are not of equal lengths.')
        # attributes describing the bandit
        self.means_list = means_list
        if stdev_list is None:
            stdev_list = [1] * len(means_list)
        self.stdev_list = stdev_list
        # attributes describing state of knowledge
        self.q_values = [0] * len(means_list)
        self.action_count = [0] * len(means_list)
        # attributes describing a learning history
        self.action_history = []
        self.reward_history = []
        self.q_history = [[0] * len(means_list)]

    @staticmethod
    def _update_average(old_average, num_obs, new_obs):
        """
        Helper function for online averaging.
        """
        # online averaging formula
        return old_average * num_obs / (num_obs + 1) + new_obs / (num_obs + 1)

    def _num_arms(self):
        """
        Returns the number of arms.
        """
        return len(self.means_list)

    def reset(self, reset_q_values: bool = True):
        """
        Resets all history attributes, and potentially also the q-values and action-counts, of the object in-place.
        """
        self.action_history = []
        self.reward_history = []
        self.q_history = [[0] * len(self.means_list)]
        if reset_q_values:
            self.q_values = [0] * len(self.means_list)
            self.action_count = [0] * len(self.means_list)

    def reward(self, arm: int):
        """
        Returns the reward of the respective arm.
        Does not modify the object.
        """
        if arm < 0 or arm > self._num_arms() - 1:
            raise ValueError("This arm does not exist.")
        from numpy.random import normal
        return normal(self.means_list[arm], self.stdev_list[arm])

    def eps_greedy_action(self, epsilon: float):
        """
        Returns the index of an arm chosen according to epsilon-greedy action-choice with respect to the q-values of the object.
        Does not modify the object.
        """
        if epsilon < 0 or epsilon > 1:
            raise ValueError("Epsilon is not in the interval [0,1].")

        # With probability epsilon
        if random.uniform(0, 1) < epsilon:
            # Chooses random action
            chosen_action = random.randint(0, self._num_arms() - 1)
        # With probability 1 - epsilon
        else:
            # Chooses action with max q-value
            chosen_action = self.q_values.index(max(self.q_values))

        return chosen_action

    def ucb_action(self, exp_factor: float = 1):
        """
        Returns the index of an arm chosen according to UCB action-choice with respect to the q-values and action-counts of the object.
        Does not modify the object.
        """
        if exp_factor < 0:
            raise ValueError("The exploration factor can not be negative.")

        # Calculates the upper bound for each arm
        upper_bounds = []
        t = sum(self.action_count) + 1
        for arm in range(self._num_arms()):
            upper_bound = self.q_values[arm] + exp_factor * math.sqrt(math.log(t) / max(1, self.action_count[arm])) # If action_count = 0, divides by 1
            upper_bounds.append(upper_bound)

        # Chooses the action with the highest upper bound
        chosen_action = upper_bounds.index(max(upper_bounds))
        return chosen_action

    def play_eps_greedy(self, num_steps=100, epsilon=0.1):
        """
        Simulates the agent playing with an epsilon-greedy policy.
        """
        # Initialization
        self.reset(True)

        # Chooses action based on epsilon-greedy policy for each step
        for step in range(num_steps):
            # Chooses action and obtains its reward
            chosen_action = self.eps_greedy_action(epsilon)
            reward = self.reward(chosen_action)

            # n <- n(a) + 1
            self.action_count[chosen_action] += 1
            # q(a) <- q(a) + 1/n(a) * (reward - q(a))
            self.q_values[chosen_action] = self._update_average(self.q_values[chosen_action], self.action_count[chosen_action], reward)
            # Updates histories
            self.action_history.append(chosen_action)
            self.reward_history.append(reward)
            self.q_history.append(self.q_values.copy())

    def play_ucb(self, num_steps=100, exp_factor=1):
        """
        Simulates the agent playing with an upper confidence bounds policy.
        """
        # Initialization
        self.reset(True)

        # Chooses action based on usb policy for each step
        for step in range(num_steps):
            # Chooses action and obtains its reward
            chosen_action = self.ucb_action(exp_factor)
            reward = self.reward(chosen_action)

            # n <- n(a) + 1
            self.action_count[chosen_action] += 1
            # q(a) <- q(a) + 1/n(a) * (reward - q(a))
            self.q_values[chosen_action] = self._update_average(self.q_values[chosen_action],
                                                                self.action_count[chosen_action], reward)
            # Updates histories
            self.action_history.append(chosen_action)
            self.reward_history.append(reward)
            self.q_history.append(self.q_values.copy())

    def plot_action_history(self, title: str = ""):
        """
        Displays a graphical representation of the action-history to standard output.
        The fraction of times an action has been chosen in the past is displayed.
        """
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.title(title)
        plt.xlabel("Time step")
        plt.ylabel("Fraction of action in past")
        for i in set(self.action_history):
            actions_chosen = [x == i for x in self.action_history]
            actions_counts = [sum(actions_chosen[0:j + 1]) for j in range(len(actions_chosen))]
            actions_percent = [actions_counts[j] / (j + 1) for j in range(len(actions_counts))]
            plt.plot(actions_percent)
        plt.show()

    def plot_q_history(self, title: str = ""):
        """
        Displays a graphical representation of the q-value-history to standard output.
        """
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.title(title)
        plt.xlabel("Time step")
        plt.ylabel("Q-Value")
        plt.plot(self.q_history)
        plt.show()

    def plot_reward_history(self, title=""):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.title(title)
        plt.xlabel("Time step")
        plt.ylabel("Reward")
        plt.scatter(range(len(self.reward_history)), self.reward_history, s=0.3)
        plt.show()


# Numerical experimentation
bandit_1 = GaussianBandit([1, 2, 3], stdev_list=[3, 2, 1])
# Epsilon Greedy
bandit_1.play_eps_greedy()
bandit_1.plot_action_history("Epsilon-Greedy(small e), Actions")
bandit_1.plot_q_history("Epsilon-Greedy(small e), Q")
bandit_1.plot_reward_history("Epsilon-Greedy(small e), Rewards")
bandit_1.play_eps_greedy(epsilon=0.9)
bandit_1.plot_action_history("Epsilon-Greedy(big e), Actions")
bandit_1.plot_q_history("Epsilon-Greedy(big e), Q")
bandit_1.plot_reward_history("Epsilon-Greedy(big e), Rewards")
# UCB
bandit_1.play_ucb()
bandit_1.plot_action_history("UCB(small exp), Actions")
bandit_1.plot_q_history("UCB(small exp), Q")
bandit_1.plot_reward_history("UCB(small exp), Rewards")
bandit_1.play_ucb(exp_factor=10)
bandit_1.plot_action_history("UCB(big exp), Actions")
bandit_1.plot_q_history("UCB(big exp), Q")
bandit_1.plot_reward_history("UCB(big exp), Rewards")

# More exploration -> All graphs have more variety, rewards are more spread out
# More exploitation -> Focused on finding best action, rewards are more often in the upper half

# Epsilon Greedy: balance between exploration and exploitation, simple
# UCB: more targeted exploration (also less-explored actions)
