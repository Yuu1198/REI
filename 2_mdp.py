import matplotlib.pyplot as plt
import numpy as np
import random


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
    policy : function
        This (possibly propabilistic) map takes a state S to an action A.
        A must be in 'actions'.
        This map must be defined for every S in 'states'.
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
    """

    def __init__(self, states, actions, transition_map=None, policy=None):
        # attributes describing the mdp
        self.states = states
        self.actions = actions
        self.transition_map = transition_map
        # attributes describing the current state and policy
        self.policy = policy
        self.state = None
        self.reward = None
        # attributes describing the history
        self.reward_history = []
        self.action_history = []
        self.state_history = []

    @staticmethod
    def _update_average(old_average, num_obs, new_obs):
        """
        Helper function for online averaging.
        """
        # online averaging formula
        return old_average * num_obs / (num_obs + 1) + new_obs / (num_obs + 1)

    def reset(self):
        """
        Resets all history attributes, and initializes the state.
        """
        from random import choice
        self.state = choice(self.states)
        self.reward_history = []
        self.action_history = []
        self.state_history = []

    def run(self, timesteps: int = 100):
        """
        Uses the policy 'timesteps' times.
        """
        for i in range(timesteps):
            action = self.policy(self.state)
            self.take_action(action)

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

    def plot_reward_history(self, title: str = ""):
        """
        Displays a graphical representation of the reward-history to standard output.
        The average reward of the past is displayed.
        """
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


# Rock-paper-scissors
states = ['R', 'P', 'S']
actions = ['R', 'P', 'S']


def rps_random_policy(state):
    """
        Chooses action with equal probability.
    """
    return np.random.choice(actions)


def rps_good_policy(state):
    """
        Choose action which would have lost in the current state.
    """
    if state == 'R':
        return 'S'
    elif state == 'P':
        return 'R'
    else:
        return 'P'


def rps_transitions(state, action):
    """
        Simulates opponents next move (updates state) and gets reward for action in previous state.
    """
    next_state = get_next_state(state)
    reward = get_reward(next_state, action)
    return next_state, reward


good_rps_mdp = MDP(states=states, actions=actions, transition_map=rps_transitions, policy=rps_good_policy)
random_rps_mdp = MDP(states=states, actions=actions, transition_map=rps_transitions, policy=rps_random_policy)


def get_next_state(state):
    """
        Gets next environment state based on opponents choice: The opponent never repeats a state.
        Otherwise, the probabilities to choose a state are equal.
    """
    available_states = []
    for s in states:
        if s != state:
            available_states.append(s)
    return np.random.choice(available_states)


def get_reward(state, action):
    """
        Gets reward based on current state and taken action.
    """
    if state == action:
        return 0
    elif ((state == 'R' and action == 'P') or
          (state == 'P' and action == 'S') or
          (state == 'S' and action == 'R')):
        return 1
    else:
        return -1


good_rps_mdp.reset()
good_rps_mdp.run(200)
good_rps_mdp.plot_action_history("Good Actions")
good_rps_mdp.plot_reward_history("Good Rewards")
good_rps_mdp.plot_state_history("Good States")

random_rps_mdp.reset()
random_rps_mdp.run(200)
random_rps_mdp.plot_action_history("Random Actions")
random_rps_mdp.plot_reward_history("Random Rewards")
random_rps_mdp.plot_state_history("Random States")
