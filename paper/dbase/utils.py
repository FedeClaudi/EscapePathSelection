from fcutils.file_io.io import load_yaml
import matplotlib.pyplot as plt

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