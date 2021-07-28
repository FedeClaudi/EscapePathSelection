from dataclasses import dataclass
import numpy as np

from rl.models.Q import QLearner
from rl.environment.environment import Status

@dataclass
class Action:
    name: str
    number: int
    is_primitive: bool = False

class Tasks():
    def __init__(self):
        self.tasks = {
            'left': Action('left', 0, is_primitive=True),
            'right': Action('right', 1, is_primitive=True),
            'up': Action('up', 2, is_primitive=True),
            'down': Action('down', 3, is_primitive=True),
            
            'threat_platform': Action('threat_platform', 4),
            'left_platform': Action('left_platform', 5),
            'right_platform': Action('right_platform', 6),
            'shelter_platform': Action('shelter_platform', 7),

            'root': Action('root', 8),
        }

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.tasks[item]
        elif isinstance(item, Action):
            return self.tasks[item.name]
        else:
            return list(self.tasks.values())[item]

    def __len__(self):
        return len(self.tasks)


class MAXQModel(QLearner):
    '''
        Implementation of the MAXQ algorithm for hierarchical model based RL.
        The implementation is inspired by: https://github.com/Kirili4ik/HRL-taxi/blob/master/Taxi.py

        For simplicity we assume that the agent has some subtaks which are already defined:
            - navigate: learn to go to a goal state using the primitive actions
            - threat, left, right, shelter: go to the corresponding platform using navigate
            - root: solve the maze navigation problem using 
    '''
    def __init__(self, game, name='MAXQ', *args, **kwargs):
        super().__init__(game, name='MAXQ', **kwargs)

        # define actions/tasks
        self.tasks = Tasks()

        # Define the task graph specifying which action are available for each subtask
        primitives = (
                    self.tasks['left'],
                    self.tasks['right'],
                    self.tasks['up'],
                    self.tasks['down']
        )

        self.graph = {
            'left': {},  # primitive actions can't use any other actions
            'right': {},
            'up': {},
            'down': {},
            
            # 'navigate' style tasks, tp dofferemt ;pcatopms
            'threat_platform': primitives,
            'left_platform': primitives,
            'right_platform': primitives,
            'shelter_platform': primitives,

            'root': (
                self.tasks['threat_platform'],
                self.tasks['left_platform'],
                self.tasks['right_platform'],
                self.tasks['shelter_platform'],
            
            ),
        }

        # define value and completion function
        n_tasks = len(self.tasks)
        self.V = np.zeros((n_tasks, self.environment.n_cells))
        self.V_copy = self.V.copy()
        self.C = np.zeros((n_tasks, self.environment.n_cells, n_tasks))

        # initialize variables
        self.done = False  # check when the task is completed

    def train(self, episodes=None, **kwargs):
        episodes = episodes or self.episodes

        for episode in range(episodes):
            # reset agent and environment
            self.reset()

            # start at the threat platform with root task
            self.MAXQ(self.tasks['root'].number, self.state_index)

            print(f'Finished episode {episode}')

    def MAXQ(self, action_number, state_index):
        '''
            MAXQ algorihtm

            Arguments:
                action_number: int, number of the selected action
                state_idnex: int. Index of the current state
        '''
        action = self.tasks[action_number]
        if not self.done:
            if action.is_primitive:
                # just take the action and learn from it
                next_state, reward, status = self.environment.step(action.number)
                self.next_state_index = self.state_indices[next_state]

                self.V[action_number, state_index] += self.learning_rate * (reward - self.V[action_number, state_index])

                if status in (Status.LOSE, Status.WIN):
                    self.done = True

                return 1
            else:
                # hierarchically execute tasks and subtasks
                n_steps = 0
                while not self.is_terminal(action_number) and not self.done:
                    # select a new action
                    action_idx = self.select_action_greedy(action_number, state_index)
                    action = self.tasks[action_idx]


                    # execute subroutines
                    N = self.MAXQ(action.number, state_index)

                    # evaluate
                    self.V_copy = self.V.copy()
                    result = self.evaluate(action_number, self.next_state_index)

                    # completion function
                    self.C[action_number, state_index, action.number] = self.learning_rate * \
                        (self.discount**N * result - self.C[action_number, state_index, action.number])

                    # update variables
                    state_index = self.next_state_index
                    n_steps += N
                return n_steps

    def reset(self):
        self.done = False
        state = self.environment.reset()  # reset and start at threat platform
        self.state_index = self.state_indices[state]

    def is_terminal(self, action_number):
        '''
            check if the current action is at a terminal state
        '''
        action = self.tasks[action_number]

        if action.is_primitive:
            return True
        elif action.name == 'root':
            return self.done
        elif action.name == 'threat_platform':
            return self.state_lookup[self.state_index] == (23, 32)
        elif action.name == 'left_platform':
            return self.state_lookup[self.state_index] == (9, 15)
        elif action.name == 'right_platform':
            return self.state_lookup[self.state_index] == (32, 23)
        elif action.name == 'shelter_platform':
            return self.state_lookup[self.state_index] == self.environment.shelter_cell

    def evaluate(self, action_number, state_index):
        action = self.tasks[action_number]
        if action.is_primitive:
            return self.V_copy[action.number, state_index]
        else:
            Q = np.arange(0)
            for sub_action in self.graph[action.name]:
                self.V_copy[sub_action.number, state_index] = self.evaluate(sub_action.number, state_index)
                Q = np.concatenate((Q, [self.V_copy[sub_action.number, state_index]]))
            max_arg = np.argmax(Q)
            return self.V_copy[max_arg, state_index]

    def select_action_greedy(self, action_number, state_index):
        epsilon = 0.001
        Q = np.arange(0)
        possible_actions = np.arange(0)

        action = self.tasks[action_number]
        for sub_action in self.graph[action.name]:
            if (
                sub_action.is_primitive or
                (not self.is_terminal(sub_action)) or
                not self.done
            ):
                Q = np.concatenate((
                    Q, 
                    [self.V[sub_action.number, state_index]] + self.C[action.number, state_index, sub_action.number]
                ))
                possible_actions = np.concatenate((possible_actions, [sub_action.number]))

        if np.random.rand(1) < epsilon:
            return np.random.choice(possible_actions)
        else:
            return possible_actions[np.argmax(Q)]
