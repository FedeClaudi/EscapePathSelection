RANDOM_INIT_POS = True  # if true during learning start position is random (when not following mice)

TRACKING_TAKE_ALL_ACTIONS = False  # if true when training on tracking data the agent
    # attempts all actions (not options) at each state. 

MAX_N_STEPS = 25  # consider a sucess only when shelter is reached in fewer steps


# TRAINING_SETTINGS = dict(
#     discount=.9999, 
#     exploration_rate=.4, 
#     learning_rate=.3,
#     episodes=100,
#     max_n_steps = 500,
# )
TRAINING_SETTINGS = dict(
    discount=.01, 
    exploration_rate=.4, 
    learning_rate=.8,
    episodes=100,
    max_n_steps = 500,
)

REWARDS = dict(
    reward_exit = 1.0,  # reward for reaching the exit cell
    penalty_move = 0,  # penalty for a move which did not result in finding the exit cell
    penalty_visited = 0,  # penalty for returning to a cell which was visited earlier
    penalty_impossible_move = -.1,  # penalty for trying to enter an occupied cell or moving out of the maze
    reward_euclidean = 0,
    reward_geodesic = 0,
)


