import sys

if sys.platform == 'darwin':
    plots_dir = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Writings/BehavPaper/plots"
    cache_dir = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Writings/BehavPaper/cache"
    flip_flop_metadata_dir = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/analysis_metadata/flipflop"
    shortcut_notes = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/analysis_metadata/shortctu/notes.yml"

    mice_records = '/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/FC_animals_records.xlsx'
    exp_records = '/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/datalog.xlsx'
else:

    # ------------------------------- dbase filling ------------------------------ #
    mice_records = 'D:\\Dropbox (UCL - SWC)\\Rotation_vte\\FC_animals_records.xlsx'
    exp_records = 'D:\\Dropbox (UCL - SWC)\\Rotation_vte\\datalog.xlsx'

    raw_data_folder = 'W:\\swc\\branco\\Federico\\raw_behaviour\\maze'
    raw_metadata_folder = 'metadata'   # appended to raw data folder
    raw_video_folder = 'video'   # appended to raw data folder
    raw_to_sort = 'to_sort'   # appended to raw data folder
    raw_analoginput_folder = 'analoginputdata'
    tracked_data_folder = 'W:\\swc\\branco\\Federico\\raw_behaviour\\maze\\pose'
    raw_ai_folder = 'W:\\swc\\branco\\Federico\\raw_behaviour\\maze\\analoginputdata'

    # trials_clips = 'W:\\swc\\branco\\Federico\\raw_behaviour\\maze\\trials_clips'   # appended to raw data folder
    trials_clips = 'W:\\swc\\branco\\Federico\\raw_behaviour\\maze\\test_clips'



    # raw_data_folder = 'K:\\TEMP'
    # # raw_metadata_folder = 'metadata'   # appended to raw data folder
    # raw_metadata_folder = 'metadata'
    # raw_video_folder = 'F:\\video'  # 'video'   # appended to raw data folder
    # raw_to_sort = 'to_sort'   # appended to raw data folder
    # raw_analoginput_folder = 'analoginputdata'
    # tracked_data_folder = 'K:\\TEMP\\pose'
    # raw_ai_folder = 'M:\\TEMP\\analoginputdata'

    # # trials_clips = 'Z:\\swc\\branco\\Federico\\raw_behaviour\\maze\\trials_clips'   # appended to raw data folder
    # trials_clips = 'Z:\\swc\\branco\\Federico\\raw_behaviour\\maze\\test_clips'



    dlc_config = 'D:\\Dropbox (UCL - SWC)\\Rotation_vte\\DLC_nets\\Nets\\Maze-Federico-2018-11-24\\config.yaml'

    commoncoordinatebehaviourmatrices = 'D:\\Dropbox (UCL - SWC)\\Rotation_vte\\CommonCoordinateBehaviourMatrix'


    # ---------------------------------- Saving ---------------------------------- #

    plots_dir = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\Writings\\BehavPaper\\plots"
    cache_dir = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\Writings\\BehavPaper\\cache"
    flip_flop_metadata_dir = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\escape\\analysis_metadata\\flipflop"
    shortcut_notes = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\shortctu\\notes.yml"