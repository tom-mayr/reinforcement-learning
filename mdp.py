import gym
from gym import spaces
import numpy as np
import random
import random

# Environment with gym library
class GymFarmEnv(gym.Env):
    """Class for the project environment"""
    
    def __init__(self, alpha=0.1, beta=0.5, max_sheep=100, max_money=1000, max_years=30, space='discrete'):
        """
        Initialize an instance of the environment.

        Args:
        alpha (float): Probability of storm destroying the wheat harvest.
        beta (float): Probability that each sheep pair produces a new sheep.
        max_sheep (int): Maximum amount of sheep in the observation space.
        max_money (int): Maximum amount of sheep in the observation space.
        max_years (int): Maximum length of an episode and maximum years in the observation space.
        space (str): How to model the observation space, either 'discrete' or 'continuous'.
        """
        super(GymFarmEnv, self).__init__()
        
        # Environment parameters
        self.alpha = alpha
        self.beta = beta
        
        # Action space: 0 = Buy one sheep, 1 = Grow wheat
        self.action_space = spaces.Discrete(2)

        # If we want discretize our space
        # Define bins for each variable
        self.n_sheep_bins = [0, 1, 2, 3, 4, 5, 10]
        self.money_bins = [0, 1000, 2000, 3000]
        self.year_bins = [0, 1, 2, 3, 4, 5, 10, 20, 30]
       
        # Observation space: Number of sheep, money, year
        # Define reasonable bounds for observation space variables
        self.max_sheep = max_sheep
        self.max_money = max_money
        self.max_years = max_years
        
        self.reset()

    def reset(self, discretize=False):
        """Reset the environment."""
        self.n_sheep = 0
        self.money = 2000
        self.year = 0
        self.done = False
        self.truncated = False
        return self._get_obs(discretize=discretize)
    
    def _get_obs(self, normalize=False, discretize=False):
        """Returns the current observation."""

        # Return normalized observation
        if normalize:
            return {
                'n_sheep': min(self.n_sheep / self.max_sheep, self.max_sheep),
                'money': min(self.money / self.max_money, self.max_money),
                'year': min(self.year / self.max_years, self.max_years)
            }
        elif discretize:
            # Digitize values into bins
            money_bin = np.digitize([self.money], self.money_bins)[0] - 1  # Subtract 1 for 0-indexing
            n_sheep_bin = np.digitize([self.n_sheep], self.n_sheep_bins)[0] - 1
            year_bin = np.digitize([self.year], self.year_bins)[0] - 1
            return {
                'n_sheep_bin': n_sheep_bin,
                'money_bin': money_bin,
                'year_bin': year_bin
            }
        else:
            return {
                'n_sheep': min(self.n_sheep, self.max_sheep),
                'money': min(self.money, self.max_money),
                'year': min(self.year, self.max_years)
            }
    
    def step(self, action, discretize=False):
        """Take a step in the environment given an action."""
        
        # Action Cost
        if action == 0:  # Buy one sheep
            cost = 1000
        else:  # Grow wheat
            cost = 20
        
        # Check if bankrupt after action cost
        if self.money - cost <= 0:
            reward = self.money - cost
            # No negative money (debt) possible for simplification
            self.money = 0
            self.done = True
            self.truncated = True
            # Advance to next year
            self.year += 1
            return self._get_obs(discretize=discretize), reward, self.done, self.truncated, {}
        
        # Decrease money by action cost
        self.money -= cost

        # Increase sheep count
        if action == 0:
            self.n_sheep += 1

        # Wool Sales
        wool_income = self.n_sheep * 10
        self.money += wool_income
        
        # Wheat Harvest
        wheat_income = 0
        if action == 1:  # Grew wheat and could afford it
            if random.random() > self.alpha:
                wheat_income = 50
                self.money += wheat_income
            else:
                # Storm destroyed the harvest
                pass  # No income from wheat
        
        # Sheep Reproduction
        if self.n_sheep > 1:
            num_pairs = self.n_sheep // 2
            new_sheep = sum(random.random() < self.beta for _ in range(num_pairs))
            self.n_sheep += new_sheep
        
        # Intermediate reward could modeled as net income for the year minus costs
        # reward = wool_income + wheat_income - cost
        
        # Advance to next year
        self.year += 1
        
        # Check termination conditions
        if self.money <= 0 or self.year >= self.max_years:
            self.done = True
            reward = self.money
        else:
            reward = 0
        
        # Prepare observation
        obs = self._get_obs(discretize=discretize)
        
        # Info dictionary can include additional data
        info = {
            'wool_income': wool_income,
            'wheat_income': wheat_income,
            'new_sheep': new_sheep if self.n_sheep > 1 else 0,
            'action_cost': cost
        }
        
        return obs, reward, self.done, self.truncated, info

    def render(self, action, reward, info, mode='human'):
        """Render some information about the current step."""
        print(f"Year: {self.year}, Sheep: {self.n_sheep}, Money: {self.money}")
        print(f"Action taken: {'buy_sheep' if action == 0 else 'grow_wheat'}")
        print(f"Reward: {reward}")
        print(f"Info: {info}")
        print("-" * 30)
