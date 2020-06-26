from fcutils.file_io.io import load_yaml
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def load_visual_stim_log(path):
    if not os.path.isfile(path): raise FileExistsError("Couldnt find log file: ", path)
    
    try: 
        log = load_yaml(path)
    except:
        raise ValueError("Could not load: ", path)

    # Transform the loaded data into a dict that can be used for creating a df
    temp_d = {k:[] for k in log[list(log.keys())[0]]}

    for stim_i in sorted(log.keys()):
        for k in temp_d.keys():
            try:
                val = float(log[stim_i][k])
            except:
                val = log[stim_i][k]
            temp_d[k].append(val)
    return pd.DataFrame.from_dict(temp_d).sort_values("stim_count")

# For manual tables
def insert_entry_in_table(dataname, checktag, data, table, overwrite=False):
    """
    dataname: value of indentifying key for entry in table
    checktag: name of the identifying key ['those before the --- in the table declaration']
    data: entry to be inserted into the table
    table: database table
    """
    if dataname in list(table.fetch(checktag)):
            return
    try:
        table.insert1(data)
        print('     ... inserted {} in table'.format(dataname))
    except:
        if dataname in list(table.fetch(checktag)):
                print('Entry with id: {} already in table'.format(dataname))
        else:
            print(table)
            raise ValueError('Failed to add data entry {}-{} to {} table'.format(checktag, dataname, table.full_table_name))


def correct_tracking_data(uncorrected, M, ypad, xpad, exp_name, sess_uid):

	"""[Corrects tracking data (as extracted by DLC) using a transform Matrix obtained via the CommonCoordinateBehaviour
		toolbox. ]
	Arguments:
		uncorrected {[np.ndarray]} -- [n-by-2 or n-by-3 array where n is number of frames and the columns have X,Y and Velocity ]
		M {[np.ndarray]} -- [2-by-3 transformation matrix: https://github.com/BrancoLab/Common-Coordinate-Behaviour]
	Returns:
		corrected {[np.ndarray]} -- [n-by-3 array with corrected X,Y tracking and Velocity data]
	"""     
	# Do the correction 
	m3d = np.append(M, np.zeros((1,3)),0)
	pads = np.vstack([[xpad, ypad] for i in range(len(uncorrected))])
	padded = np.add(uncorrected, pads)
	corrected = np.zeros_like(uncorrected)

	# ? SLOW
	"""
		x, y = np.add(uncorrected[:, 0], xpad), np.add(uncorrected[:, 1], ypad)  # Shift all traces correctly based on how the frame was padded during alignment 
		for framen in range(uncorrected.shape[0]): # Correct the X, Y for each frame
			xx,yy = x[framen], y[framen]
			corrected[framen, :2] = (np.matmul(m3d, [xx, yy, 1]))[:2]
	"""

	# ! FAST
	# affine transform to match model arena
	concat = np.ones((len(padded), 3))
	concat[:, :2] = padded
	corrected = np.matmul(m3d, concat.T).T[:, :2]

	# Flip the tracking on the Y axis to have the shelter on top
	midline_distance = np.subtract(corrected[:, 1], 490)
	corrected[:, 1] = np.subtract(490, midline_distance)
	return corrected


def get_arm_given_rois(rois, direction):
        """
            Get arm of origin or escape given the ROIs the mouse has been in
            direction: str, eitehr 'in' or 'out' for outward and inward legs of the trip
        """
        rois_copy = rois.copy()
        rois = [r for r in rois if r not in ['t', 's']]

        if not rois:
            return None

        if direction == 'out':
            vir = rois[-1]  # very important roi
        elif direction == "in":
            vir = rois[0]

        if 'b15' in rois:
            return 'Centre'
        elif vir == 'b13':
            return 'Left2'
        elif vir == 'b10':
            if 'p1' in rois or 'b4' in rois:
                return 'Left_Far'
            else:
                return 'Left_Medium'
        elif vir == 'b11':
            if 'p4' in rois or 'b7' in rois:
                return 'Right_Far'
            else:
                return 'Right_Medium'
        elif vir == 'b14':
            return 'Right2'
        else:
            return None

def get_roi_enters_exits(roi_tracking, roi_id):
	"""get_roi_enters_exits [Get all the timepoints in which mouse enters or exits a specific roi]
	
	Arguments:
		roi_tracking {[np.array]} -- [1D array with ROI ID at each frame]
		roi_id {[int]} -- [roi of interest]
	"""

	in_roi = np.where(roi_tracking == roi_id)[0]
	temp = np.zeros(roi_tracking.shape[0])
	temp[in_roi] = 1
	enter_exit = np.diff(temp)  # 1 when the mouse enters the platform an 0 otherwise
	enters, exits = np.where(enter_exit>0)[0], np.where(enter_exit<0)[0]
	return enters, exits

def convert_roi_id_to_tag(ids):
    rois_lookup = load_yaml('paper/dbase/rois_lookup.yml')
    rois_lookup = {v:k for k,v in rois_lookup.items()}
    return [rois_lookup[int(r)] for r in ids]


def load_rois(display=False):
    components = load_yaml('paper/dbase/template_components.yml')
    rois = {}

    # Get platforms
    for pltf, (center, radius) in components['platforms'].items():
        rois[pltf] = tuple(center)

    # Get bridges
    for bridge, pts in components['bridges'].items():
        x, y = zip(*pts)
        center = (max(x)+min(x))/2., (max(y)+min(y))/2.
        rois[bridge] =  center

    if display:
        [print('\n', n, ' - ', v) for n,v in rois.items()]
    
    return rois


def plot_rois_positions():
    rois = load_rois()
    
    f, ax= plt.subplots()
    for roi, (x, y) in rois.items():
        ax.scatter(x, y, s=50)
        ax.annotate(roi, (x, y))
    ax.set(xlim=[0, 1000], ylim=[0, 1000])
    plt.show()

def get_roi_at_each_frame(experiment, session_name, bp_data, rois=None):
    """
    Given position data for a bodypart and the position of a list of rois, this function calculates which roi is
    the closest to the bodypart at each frame
    :param bp_data: numpy array: [nframes, 2] -> X,Y position of bodypart at each frame
                    [as extracted by DeepLabCut] --> df.bodypart.values
    :param rois: dictionary with the position of each roi. The position is stored in a named tuple with the location of
                    two points defyining the roi: topleft(X,Y) and bottomright(X,Y).
    :return: tuple, closest roi to the bodypart at each frame
    """

    def check_roi_tracking_plot(session_name, rois, centers, names, bp_data, roi_at_each_frame):
        save_fld = 'D:\\Dropbox (UCL - SWC)\\Rotation_vte\\Maze_templates\\ignored\\Matched'
        rois_ids = {p:i for i,p in enumerate(rois.keys())}
        roi_at_each_frame_int = np.array([rois_ids[r] for r in roi_at_each_frame])

        f, ax = plt.subplots()
        ax.scatter(bp_data[:, 0], bp_data[:, 1], c=roi_at_each_frame_int, alpha=.4)
        for roi, k in zip(centers, names):
            ax.plot(roi[0], roi[1], 'o', label=k)
        ax.legend()
        # plt.show()
        f.savefig(os.path.join(save_fld, session_name+'.png'))

    if rois is None:
        rois = load_rois()
    elif not isinstance(rois, dict): 
        raise ValueError('rois locations should be passed as a dictionary')

    if not isinstance(bp_data, np.ndarray):
            pos = np.zeros((len(bp_data.x), 2))
            pos[:, 0], pos[:, 1] = bp_data.x, bp_data.y
            bp_data = pos

    # Get the center of each roi
    centers, roi_names = [], [] 
    for name, points in rois.items():  # a point is two 2d XY coords for top left and bottom right points of roi
        points = points.values[0]
        if not isinstance(points, np.ndarray): 
            if isinstance(points, (tuple, list)):
                points= np.array(points)
            else:
                continue # maze component not present in maze for this experiment
        try:
            center_x = points[1] + (points[3] / 2)
        except:
            # raise ValueError('Couldnt find center for points: ',points, type(points))
            center_x = points[0]
            center_y = points[1]
        else:
            center_y = points[0] + (points[2] / 2)
        
        # Need to flip ROIs Y axis to  match tracking
        dist_from_midline = 500 - center_y
        center_y = 500 + dist_from_midline
        center = np.asarray([center_x, center_y])
        centers.append(center)
        roi_names.append(name)

    # Calc distance to each roi for each frame
    data_length = bp_data.shape[0]
    distances = np.zeros((data_length, len(centers)))

    for idx, center in enumerate(centers):
        cnt = np.tile(center, data_length).reshape((data_length, 2))
        dist = np.hypot(np.subtract(cnt[:, 0], bp_data[:, 0]), np.subtract(cnt[:, 1], bp_data[:, 1]))
        distances[:, idx] = dist

    # Get which roi the mouse is in at each frame
    sel_rois = np.argmin(distances, 1)
    roi_at_each_frame = tuple([roi_names[x] for x in sel_rois])

    # Check we got cetners correctly
    check_roi_tracking_plot(session_name, rois, centers, roi_names, bp_data, roi_at_each_frame)
    return roi_at_each_frame
