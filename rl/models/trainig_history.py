import numpy as np
from rl.environment.environment import Status

class TrainingHistory:
    '''
        Keep Track of variables during training.
        Useful for model comparison
    '''
    
    def __init__(self):
        # variables to track training progerss
        self.tot_training_steps = 0
        self.tot_reward = 0
        self.reward_history = []  # reward at each step during training
        self.mean_episode_reward_history = []  # mean episode reward durin training
        self.max_episode_reward_history = []
        self.episode_length_history = []  # number of steps for each episode
        self.episode_distance_history = []  # distance covered in each episode
        self.successes_history = []  # 1 if episode ends in agent at shelter else 0
        self._prev_state = None
        self.state_history = []  # keeps track of the state at each training step

        # keep track of 'play' test scores during training
        self.play_status_history = []
        self.play_steps_history = []
        self.play_reward_history = []
        self.play_arm_history = []

        self.data = {
            'tot_training_steps':self.tot_training_steps,
            'tot_reward':self.tot_reward,
            'reward_history':self.reward_history,
            'mean_episode_reward_history':self.mean_episode_reward_history,
            'max_episode_reward_history':self.max_episode_reward_history,
            'episode_length_history':self.episode_length_history,
            'episode_distance_history':self.episode_distance_history,
            'successes_history':self.successes_history,
            '_prev_state':self._prev_state,
            'play_status_history':self.play_status_history,
            'play_steps_history':self.play_steps_history,
            'play_reward_history':self.play_reward_history,
            'play_arm_history':self.play_arm_history,
        }

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'''Training history:\n
            tot_training_steps: {self.tot_training_steps}\n
            tot_reward: {self.tot_reward}\n
            reward_history: {self.reward_history}\n
            mean_episode_reward_history: {self.mean_episode_reward_history}\n
            episode_length_history: {self.episode_length_history}\n'''

    def on_episode_start(self):
        self._episode_steps = 0
        self._episode_rewards = []
        self._distance_travelled = 0

    def on_step_end(self, reward, state):
        self.tot_training_steps += 1
        self._episode_steps += 1

        self.tot_reward += reward
        self.reward_history.append(reward)
        self._episode_rewards.append(reward)

        # measure distance travelled
        state = np.array(state)
        if self._prev_state is not None:
            self._distance_travelled += np.linalg.norm(state - self._prev_state)
        
        self.state_history.append(state)
        self._prev_state = state

    def update_play_history(self, status, step, reward, arm):
        self.play_status_history.append(1 if status == Status.WIN else 0)
        self.play_steps_history.append(step)
        self.play_reward_history.append(reward)
        self.play_arm_history.append(arm)

    def on_episode_end(self, status):
        self.max_episode_reward_history.append(np.max(self._episode_rewards))
        self.mean_episode_reward_history.append(np.mean(self._episode_rewards))
        self.episode_length_history.append(self._episode_steps)
        self.episode_distance_history.append(self._distance_travelled)

        if status == Status.WIN:
            self.successes_history.append(1)
        else:
            self.successes_history.append(0)
