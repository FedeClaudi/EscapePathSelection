import sys
sys.path.append('./')

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from random import choice
import pandas as pd
import random
import pickle
from scipy.special import softmax
from tqdm import tqdm 

from Modelling.maze_solvers.agent import Agent
from Utilities.maths.math_utils import calc_distance_between_points_2d as dist


class VanillaMB(Agent):
	def __init__(self, grid_size=None, **kwargs):
		Agent.__init__(self, grid_size=grid_size, **kwargs)

		self.block_probability = 0.0  # ? probabilistic block in state transition prob func

		self.value_est_iters = 50

		self.P = self.define_transitions_func()    			# transition function
		self.R = np.zeros(len(self.free_states))		# Reward function
		self.V = np.zeros(len(self.free_states))		# Value function

		self.n_steps = 10000

	def run(self):
		# prep figure
		f, axarr =  plt.subplots(ncols=5, nrows=3)

		# Learn the reward and transitions functions
		exploration = self.explore()

		# Perform value estimation
		self.value_estimation()
		self.plot_func("P", ax=axarr[0, 0], ttl="MB - Naive")
		self.plot_func("V", ax=axarr[1, 0], ttl="")

		# Do a walk and plot
		walk = self.walk_with_state_transitions()
		self.plot_walk(walk, ax=axarr[2, 0])

		if self.maze_type == "asymmetric":
			self.introduce_blockage("right")
			self.reset_values()
			self.value_estimation()
			walk = self.walk_with_state_transitions()
			self.plot_func("P", ax=axarr[0, 1], ttl="Blocked LAMBDA")
			self.plot_func("V", ax=axarr[1, 1], ttl="")
			self.plot_walk(walk,ax=axarr[2, 1])
		else:
			# Introduce blocage on LAMBDA
			self.introduce_blockage('lambda')
			
			# Recompute values and do a walk
			self.reset_values()
			self.value_estimation()
			
			walk = self.walk_with_state_transitions()
			self.plot_func("P", ax=axarr[0, 1], ttl="Blocked LAMBDA")
			self.plot_func("V", ax=axarr[1, 1], ttl="")
			self.plot_walk(walk,ax=axarr[2, 1])

			# Do one trial start_locationing at P with LAMBDA *fully* closed
			self.introduce_blockage('lambda', p=0)

			self.reset_values()
			self.value_estimation()
			walk = self.walk_with_state_transitions(start_location="secondary")
			self.plot_func("P", ax=axarr[0, 4], ttl="start_location at P")
			self.plot_func("V", ax=axarr[1, 4], ttl="")
			self.plot_walk(walk,ax=axarr[2, 4])


			# Relearn state transitions (reset to before blockage)
			self._reset()
			self.explore()

			# Block alpha and do a walk
			self.introduce_blockage("alpha1")
			self.reset_values()
			self.value_estimation()
			walk = self.walk_with_state_transitions()

			self.plot_func("P", ax=axarr[0, 2], ttl="Blocked - ALPHA1")
			self.plot_func("V", ax=axarr[1, 2], ttl="")
			self.plot_walk(walk, ax=axarr[2, 2])

			# Reset and repeat with both alphas closed
			self._reset()
			self.explore()
			self.introduce_blockage("alpha")
			self.reset_values()
			self.value_estimation()
			walk = self.walk_with_state_transitions()

			self.plot_func("P", ax=axarr[0, 3], ttl="Blocked - ALPHAs")
			self.plot_func("V", ax=axarr[1, 3], ttl="")
			self.plot_walk(walk, ax=axarr[2, 3])

			self._reset()

	def _reset(self):
		self.free_states = self._free_states.copy()
		self.maze = self._maze.copy()

	def reset_values(self):
		self.V = np.zeros(len(self.free_states))	

	def plot_func(self, func="R", ax=None, ttl = None):
		if ax is None:
			f, ax = plt.subplots()

		if func == "R":
			policy = self.R
			title = ttl + " - reward function"
		elif func=='V': 
			title = ttl + " - value function"
			policy = self.V
		else:
			title = ttl + " - transition function"
			policy = self.P.copy()
			policy[policy == 0] = np.nan
			# policy = np.sum(self.P, 1)[:, 0]
			policy = np.nanmin(np.nanmin(policy, 2),1)
		# Create an image
		img = np.full(self.maze.shape, np.nan)
		for i, (x,y) in enumerate(self.free_states):
			img[x,y] = policy[i]

		# ax.scatter([x for x,y in self.free_states], [y for x,y in self.free_states], c=policy)
		ax.imshow(np.rot90(img, 3)[:, ::-1])
		ax.set(title = title, xticks=[], yticks=[])

	def define_transitions_func(self):
		return np.zeros((len(self.free_states), len(self.free_states), len(self.actions))).astype(np.float16)

	def explore(self):
		curr = self.start_location.copy()
		walk = []

		for step in np.arange(self.n_steps):
			walk.append(curr.copy())

			nxt, action_n = self.step("shelter", curr) # ? take random step

			# update state transition function
			curr_index  = self.get_state_index(curr)
			nxt_index  = self.get_state_index(nxt)

			if nxt == self.goal_location:
				self.R[nxt_index] = 1

			self.P[curr_index, nxt_index, action_n] = 1  # set as 1 because we are in a deterministic world

			curr = nxt

		walk.append(curr)
		return walk
	
	def estimate_state_value(self, state):
		# Get the reward at the current state
		idx = self.get_state_index(state)
		reward = self.R[idx]

		# Get which actions can be perfomed and the values of the states they lead to
		valid_actions = self.get_valid_actions(state)
		landing_states_values = [self.V[si] for si, a, p in valid_actions] #  get the values of the landing states to guide the prob of action selection
		if not np.any(landing_states_values): 
			action_probs = [1/len(valid_actions) for i in np.arange(len(valid_actions))]  # if landing states have no values choose each random with same prob
		else:
			action_probs = softmax(landing_states_values)

		"""
			landing_states_values = [self.V[si] for si, a, p in valid_actions]

			if landing_states_values:
				# If the landing states don't have values, choose a random one
				if not np.max(landing_states_values) == 0:
					action_prob = softmax(landing_states_values)   # ? policy <- select highest value option with higher freq
				else:
					# Each action has equal probability of bein selected
					action_prob = [1/len(valid_actions) for i in np.arange(len(valid_actions))]

				# The value of the current state is given by the sum for each action of the product of:
				# the probability of taking the action
				# the value of the landing state
				# the probaility of taking getting to the state (transtion function)

				value = np.sum([action_prob[i] * self.V[s1] * p for i, (s1,a, p) in enumerate(valid_actions)])
				self.V[idx] = reward + value
		"""
		value = 0 # initialise the value as 0
		if valid_actions:
			for (s1_idx, action, transition_prob), action_prob in zip(valid_actions, action_probs):
				r = self.R[s1_idx]  # reward at landing state
				# transition_prob = probability of reaching s1 given s0,a -> p(s1|s0, a)
				# action_prob = pi(a|s) -> probability of taking action a given state s and policy pi
				s1_val = self.V[s1_idx]  # value of s1

				if 0 < transition_prob < 1 and s1_val > 0:
					a =1 

				#action value: pi(a|s0)    * p(s1|s0,a)      *  [R(s)+V(s1)]
				action_value = action_prob * transition_prob *  (r + s1_val)
				value += action_value # the value is the sum across all action values

		self.V[idx] = value
		
	def value_estimation(self, ax=None):
		print("\n\nValue estimation")
		for i in tqdm(range(self.value_est_iters)):
			for state in self.free_states:
				self.estimate_state_value(state)

		if ax is not None:
			ax.scatter([x for x,y in self.free_states], [y for x,y in self.free_states], c=self.V)
			
	def get_valid_actions(self, state):
		idx = self.get_state_index(state)
		valid_actions = []
		for state_idx, state in enumerate(self.P[idx]):
			[valid_actions.append((state_idx, action, p)) for action, p in enumerate(state) if p > 0]

		return valid_actions

	def walk_with_state_transitions(self, start_location=None, probabilistic = False, avoid_states=[]):
		if start_location is None:
			curr = self.start_location.copy()
		else: 
			curr = self.second_start_location.copy()

		walk, walk_idxs = [], []

		reached_goal_location = False
		for step in np.arange(50):
			walk.append(curr)

			current_index = self.get_state_index(curr)
			walk_idxs.append(current_index)
			valid_actions = self.get_valid_actions(curr)
			
			if avoid_states: # avoid going to states visited during previous walks
				if step > 3:
					valid_actions = [(s,a,p) for s,a,p in valid_actions if s not in avoid_states and a not in [3] and s not in walk_idxs]
					if not valid_actions: break

			values = [self.V[si] for si,a,p in valid_actions]

			if not probabilistic: # choose the action lading to the state with the highest value
				selected = self.free_states[valid_actions[np.argmax(values)][0]]
			else:  # choose the actions with a probability proportional to their relative value
				selected = self.free_states[random.choices(valid_actions, weights=softmax(values), k=1)[0][0]]
			curr = selected

			if reached_goal_location: break

			if curr == self.goal_location or dist(self.goal_location, curr) < 2: 
				reached_goal_location = True # do one more step and then stop

		if reached_goal_location:
			walk.append(curr)
			return walk



	def do_probabilistic_walks(self, n=100):
		visited, walks = [], []
		for i in np.arange(n):
			walk = self.walk_with_state_transitions(probabilistic=True, avoid_states=visited)
			if walk is None: continue
			else:
				visited.extend([self.get_state_index(s) for s in walk])
				walks.append(walk)

		f, ax = plt.subplots()
		ax.imshow(self.maze_image, cmap="Greys_r")
		for walk in walks:
			if walk is not None:
				self.plot_walk(walk, ax=ax, background=False, multiple=True)


	def step(self, policy, current):
		legal_actions = [a for a in self.get_available_moves(current = current) if a != "still" not in a]
		action = choice(legal_actions)

		# move
		action_n = [k for k,v in self.actions.items() if v == action][0]
		return self.move(action, current), action_n

	def introduce_blockage(self, bridge, p=0):
		blocks = self.get_blocked_states(bridge)

		actions_to_block = [0, 1, 2, 4, 5] # Only actions between left, up, right should be blocked
		
		for state in blocks:
			try:
				idx = self.get_state_index(state)
			except:
				pass
			else:
				# Get all the states-actions leading to state and set the probability to p
				self.P[:, idx] = np.where(self.P[:, idx], p, self.P[:, idx])
				
			self.maze[state[1], state[0]] = self.block_probability
			



if __name__ == "__main__":
	mb = VanillaMB()
	mb.run()
	plt.show()


