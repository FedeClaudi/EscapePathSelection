from paper.trials import TrialsLoader
import pandas as pd


# Load trials
trials = TrialsLoader(naive = None,
                        lights = None, 
                        tracking = 'all')
trials.load_psychometric()
trials.remove_change_of_mind_trials() # remove silly trials
trials.keep_catwalk_only()

# poool and save
trs = pd.concat(trials.datasets.values(), ignore_index=True)
trs.to_hdf('D:\\Dropbox (UCL)\\Rotation_vte\\Locomotion\\control\\behav_data\\psychometric_trials.h5', key='hdf')
print('done')