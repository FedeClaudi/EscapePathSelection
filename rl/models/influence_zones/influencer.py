import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
import random
from celluloid import Camera
import random

from myterial import pink, blue_light

from rl.models.Q import QLearner
from rl.models.tracking import QLearnerTracking
from rl.models.influence_zones.itm import ITM
from rl.environment.actions import Actions, Action

class InfluenceZones(QLearner):
    _predict_with_shelter_vector = False
    
    phi = 1e-10  # when TD error > phi 1-step update is triggered
    theta_max = .1  # when theta > theta_max N-step update is triggered

    value_th = .6
    
    def __init__(self, game, name='InfluenceZones', *args, film=False, predict_with_shelter_vector=None, **kwargs):
        super().__init__(game, name='InfluenceZones', **kwargs)
        self.actions = Actions()

        self.predict_with_shelter_vector = predict_with_shelter_vector or self._predict_with_shelter_vector

        self.count = 0  # counts learning steps
        self.theta = 0  # accumulates TD error

        # instantiate topological map
        self.shelter_vector_target = np.zeros(2)
        self.itm = ITM()

        # initialize shelter vector
        self.shelter_vector = None

        # create a figure to make video
        self.film = film
        if film:
            figure, self.ax = plt.subplots(figsize=(9, 9))
            self.camera = Camera(figure)

            self.ax.axis('off')
            self.ax.invert_yaxis()
            self.ax.set(title='topological  neighbors')

    @property
    def n_actions(self):
        return len(self.actions)

    def q(self, state):
        """ Get q values for all actions for a certain state. """
        node = self.itm.get_closest(state)
        q = np.array([np.dot(node.target_delta, action.shift) for action in self.actions.actions])
        q = [qq if qq/q.max() > .1 else 0 for qq in q]
        return q

    @property
    def q_max(self):
        ''' max value of any state action pair '''
        return np.max(np.vstack(n.Q for n in self.nodes))

    @property
    def q_min(self):
        ''' min value of any state action pair '''
        return np.min(np.vstack(n.Q for n in self.nodes))

    def state_q_max(self, state_index):
        '''
            Max Q value at a state
        '''
        return self.itm.get_closest(np.array(state)).V

    def training_step(self, state_index):
        '''
            Single step during training of one episode.

            Arguments:
                state_index: int. State index
        ''' 
        self.training_mode = True

        # update topological map
        state = self.state_lookup[state_index]
        self.itm.update(state)
        
        # select an action
        action = self.choose_epsilon_greedy_action(state_index)

        # execute action
        next_state, reward, status = self.environment.step(action)
        next_state_index = self.state_indices[next_state]

        # update shelter vector
        if next_state_index != state_index:
            self.update_shelter_vector(state, next_state)

        # update V and Q values
        # compute TD error for each neighbor of the current state
        current_node = self.itm.get_closest(state)
        current_node.r = reward

        for next_node in current_node.neighors:
            td_error = self._compute_td_error(current_node, next_node, reward)
            self.theta += td_error

            # 1-step update
            if td_error > self.phi:
                self._one_step_update(current_node, next_node, td_error)

        # N-step update
        if self.theta > self.theta_max:
            # reset accumulator
            self.theta = 0

            # n-steps update
            self._n_steps_update(current_node)

        return next_state_index, reward, status


    def _get_action_to_node(self, current_node, next_node, index=False):
        '''
            Gets which action is closest to the one leading 
            from the current node to the closest node

            Arguments:
                current_node: Node of start
                next_node: Node of end
                index: bool, if True the action index not vector is returned
        '''
        delta = next_node - current_node
        dots = [np.dot(action.shift, delta) for action in self.actions.actions] 

        if not index: 
            return self.actions[np.argmax(dots)]
        else:
            return np.argmax(dots)

    def predict(self, state_index=0):
        '''
            Selects the best action to move from a given state toward the node
            on the topological map with higest value
        '''
        # get target node
        if isinstance(state_index, tuple):
            state = state_index
        else:
            state = np.array(self.state_lookup[state_index])
        self.state = state

        # get ITM node closest to state
        node = self.itm.get_closest(state)

        # if the node doesn't have edges, return random action
        if not node.edges:
            return random.choice(self.environment.actions)

        # get normalized values for neighboring nodes
        maxV = np.max([n.V for n in node.edges])
        if maxV:
            values = [n.V/maxV for n in node.edges]
        else:
            values = [n.V for n in node.edges]

        # select node that maximizes both
        if self.predict_with_shelter_vector and self.shelter_vector is not None:
            # get normalized dot products wrt shelter edge
            dots = []
            for n in node.edges:
                delta = n - node
                dots.append(np.dot(delta, self.shelter_vector)/np.linalg.norm(delta))

            # choose using dots
            choice_values = [v+d if v > self.value_th else -100 for v,d in zip(values, dots)]
            target = node.edges[np.argmax(choice_values)]
        else:
            # choose based on values only
            choice_values = []
            target = self.itm._get_highest_value_node(node.edges)

        # get position delta
        delta = target - node

        # get action from node -> target
        available_actions = self.environment._possible_actions()
        action = available_actions[np.argmax(
            [np.dot(delta, action.shift) for action in available_actions]
        )]

        # update shelter vector
        self.update_shelter_vector(state, np.array(state) + action.shift)
        return action

    def _one_step_update(self, current_node, next_node, td_error):
        '''
            Performs 1-step update
        '''
        # update Q
        current_node.Q[next_node.id] += self.learning_rate * td_error

    def _compute_td_error(self, current_node, next_node, reward):
        '''
            Computes the TD error
        '''
        # check that nodes have Q
        for node in (current_node, next_node):
            # update Q
            node.update_Q()
        
        # check that there's a node change
        if next_node.id == current_node.id:
            return 0
        else:
            return reward + self.discount * next_node.V - current_node.Q[next_node.id]
        
    def _n_steps_update(self, current_node):
        '''
            Performs n-steps update
        '''
        count = 0
        # get neighborhoods
        neighborhoods = self.itm.construct_neighborhoods(current_node)

        # do updates
        for n, neighborhood in enumerate(neighborhoods):
            if n == 0:
                continue

            # loop over nodes in neighborhood
            for node in neighborhood:
                if node.V < current_node.V:
                    target = self.itm._get_highest_value_node(node.edges)

                    # get TD error
                    error_td = self._compute_td_error(node, target, target.r)

                    # update Q
                    node.Q[target.id] += self.learning_rate * error_td

                    count += 1

    def update_shelter_vector(self, state, next_state):
        '''
            Updates/resets the shelter vector
            based on the current state
        '''
        # check that the vector is correct
        # if self.shelter_vector is not None:
        #     if np.any(np.array(state) + self.shelter_vector != np.array(self.shelter_vector_target)):
        #         # raise ValueError(f'Shelter vector {self.shelter_vector} doesnt go from state {state} to shelter {self.environment.shelter_cell}')
        #         logger.warning(f'Shelter vector {self.shelter_vector} doesnt go from state {state} to shelter {self.environment.shelter_cell}')

        if self.shelter_vector is None and self.environment.shelter_found:
            self.shelter_vector_target = state
            logger.debug(f'Zeroing shelter vector at: {state}')
            self.shelter_vector = np.zeros(2)
            
        if self.shelter_vector is not None:
            shift = np.array(next_state) - np.array(state)
            # update it  vector
            if self.count > 300:
                a = 1
            self.shelter_vector -= shift



    def visualize_shelter_vector(self, ax, state):
        '''
            Visualizes the homing vector as a line starting from the current state
            and with direction/magnitude equal to the shelter vector
        '''
        if self.shelter_vector is not None:
            end = state + self.shelter_vector
            ax.plot(
                [state[0], end[0]],
                [state[1], end[1]],
                lw=2,
                color='r',
            )

    def save_video(self):
        '''
            Saves a video with learning snapshots
        '''
        if self.film:
            animation = self.camera.animate()
            logger.info('Saving influencer.mp4')
            animation.save('influencer.mp4')

    def on_reset(self, state):
        '''
            Resets shelter vector for training
        '''
        self.count = 0  # counts learning steps
        self.theta = 0  # accumulates TD error
        self.shelter_vector = None

        
    def on_play_start(self, state):
        '''
            Reset the shelter vector before starting 
            play mode
        '''
        self.shelter_vector = self.shelter_vector_target - np.array(state)

    # def on_play_end(self):
    #     '''
    #         Plot things after the playing is done
    #     '''
    #     f, ax = plt.subplots(figsize=(9, 9))
    #     f.suptitle('Final ITM')
    #     self.itm.visualize(value=True, ax=ax, annotate=False)
    #     self.visualize_shelter_vector(ax, self.state)

    #     ax.imshow(self.environment.maze, cmap='binary', alpha=.8)
    #     ax.axis('off')


    # def on_training_end(self):
    #     '''
    #         Called when done training
    #     '''
    #     if self.film:
    #         self.save_video()
    #         self.itm.visualize_3D()

    #     f, ax = plt.subplots(figsize=(9, 9))
    #     f.suptitle('Final ITM')
    #     self.itm.visualize(value=True, ax=ax, annotate=False)

    #     ax.imshow(self.environment.maze, cmap='binary', alpha=.8)
    #     ax.axis('off')

class InfluenceZonesTracking(QLearnerTracking, InfluenceZones):
    def __init__(self, environment, maze_name, trial_number=None, name='InfluenceZonesTracking', **kwargs):
        super().__init__(environment, maze_name, trial_number=trial_number, name='InfluenceZonesTracking', **kwargs)

        InfluenceZones.__init__(self, environment, **kwargs)

    
    def tracking_training_step(self, state_index, action, next_state_index, reward):
        # update topological map
        state = self.state_lookup[state_index]
        self.itm.update(state)

        # update V and Q values
        # compute TD error for each neighbor of the current state
        current_node = self.itm.get_closest(state)
        current_node.r = reward
        next_node = self.itm.get_closest(self.state_lookup[next_state_index])

        for next_node in current_node.neighors:
            td_error = self._compute_td_error(current_node, next_node, reward)
            self.theta += abs(td_error)

            # 1-step update
            if abs(td_error) > self.phi:
                self._one_step_update(current_node, next_node, td_error)

        # N-step update
        if self.theta > self.theta_max:
            # reset accumulator
            self.theta = 0

            # n-steps update
            self._n_steps_update(current_node)

            # mark that an N steps update happened
            if self.film:
                self.ax.scatter(10, 10, s=100, color='r', zorder=200)

        # update homing vector
        self.update_shelter_vector(state, self.state_lookup[next_state_index])

        # add frame to video
        if self.count % 1 == 0 and self.film:
            if self.count > 300:
                logger.debug(f'Creating animation frame {int(self.count)}')
                self.itm.visualize(value=True, ax=self.ax, state=np.array(state))
                self.visualize_shelter_vector(self.ax, state)
                self.camera.snap()
        self.count += 1

