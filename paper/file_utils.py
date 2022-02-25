import os
import sys

sys.path.append("./")

from paper import paths
import paper.video_converter as converter

def sort_mantis_files():
    # Get folders paths
    raw = paths.raw_data_folder
    metadata_fld = os.path.join(raw, paths.raw_metadata_folder)
    video_fld = os.path.join(raw, paths.raw_video_folder)
    tosort_fld = os.path.join(raw, paths.raw_to_sort)
    ai_fld = os.path.join(raw, paths.raw_analoginput_folder)

    log = open(os.path.join(tosort_fld, 'log.txt'), 'w+')
    log.write('\n\n\n\n')
    # Loop over subfolders in tosort_fld
    for fld in os.listdir(tosort_fld):
        log.write('Processing Folder {}'.format(fld))
        print('Processing Folder ', fld)
        # Loop over individual files in subfolder
        if  '.txt' in fld: continue # skip log file
        for f in os.listdir(os.path.join(tosort_fld, fld)):
            if '.txt' in f: continue # skip log file
            print('     Moving: ', f)
            # Get the new name and destination for each file
            if 'Maze' in f:
                # Its the file with the AI
                newname = fld+'.tdms'
                dest = ai_fld
            elif 'Overview' in f:
                newname = fld+'Overview.tdms'
                if 'meta' in f:
                    # Overview Camera Metadata file
                    dest = metadata_fld
                else:
                    # Overview Camera Data file
                    dest = video_fld
            elif 'Threat' in f:
                newname = fld+'Threat.tdms'
                if 'meta' in f:
                    # Threat Camera Metadata file
                    dest = metadata_fld
                else:
                    # Threat Camera Data file
                    dest = video_fld
            elif "visual_stimuli_log" in f:
                # it's the file logging all the visual stimuli delivered
                newname = fld+"visual_stimuli_log.yml"
                dest = ai_fld
            else:
                raise ValueError('Unexpected file: ', os.path.join(fld, f))

            original = os.path.join(tosort_fld, fld, f)
            moved = os.path.join(dest, newname)
            # print(f"{original} ==> {moved}")
            
            try:
                os.rename(original, moved)
                log.write('Moved {} to {}'.format(original, moved))
            except:
                print('         Didnt move file because already exists')
                log.write('!!NOT!!! Moved {} to {}'.format(original, moved))
        log.write('Completed Folder {}\n\n'.format(fld))
    log.close()



def check_if_file_converted(name, folder):
    conv, join = False, False
    mp4s = [v for v in os.listdir(folder) if name in v and '.mp4' in v]
    # print(name)
    if mp4s:
        joined = [v for v in mp4s if 'joined' in mp4s]
        conv = True
        if joined: join=True

    return conv, join

def get_list_uncoverted_tdms_videos():
    """
        Check which videos still need to be converted
    """
    videos_fld = os.path.join(paths.raw_data_folder, paths.raw_video_folder)

    tdmss = [f for f in os.listdir(videos_fld) if '.tdms' in f]
    unconverted = []
    for t in tdmss:
        name = t.split('.')[0]
        conv, join = check_if_file_converted(name, videos_fld)
        if not conv: 
            unconverted.append(t)
        
    # store names to file
    # store = "Utilities/file_io/files_to_convert.yml"
    # with open(store, 'w') as out:
    #     yaml.dump([os.path.split(u)[-1] for u in unconverted], out)


    print('To convert: ', unconverted)
    print(len(unconverted), ' files yet to convert')
    return unconverted


def convert_tdms_to_mp4():
    """
        Keeps calling video conversion tool, regardless of what happens
    """
    videos_fld = os.path.join(paths.raw_data_folder, paths.raw_video_folder)
    metadata_fld = os.path.join(paths.raw_data_folder, paths.raw_metadata_folder)

    tcvt = get_list_uncoverted_tdms_videos()
    toconvert = [os.path.join(videos_fld, t) for t in tcvt]
    # while True:

    for f in toconvert:
        metadata = os.path.join(
            metadata_fld, os.path.split(f)[1]
        )
        if not os.path.exists(metadata):
            print(f'Metdata for {f} not found, skipping')
            continue
        # get metadata file
        
        
        converter.convert(f, metadata)


        # try:
        #     VideoConverter(os.path.join(videos_fld, f), extract_framesize=True)
        # except:
        #     print(f"Failed to convert {f} !!!!!!!!")
        #     continue


    # def get_list_not_tracked_videos(self):
    #     videos = [f.split('.')[0] for f in os.listdir(self.videos_fld) if 'tdms' not in f and "." in f and not "Threat" in f]
    #     poses = [f.split('_')[:-1] for f in os.listdir(self.pose_fld) if 'h5' in f]

    #     not_tracked = []
    #     for f in videos:
    #         videoname = os.path.join(self.videos_fld, f+".mp4")
    #         if not os.path.isfile(videoname): videoname = os.path.join(self.videos_fld, f+".avi")
    #         if os.path.getsize(videoname) > 10000 and f.split('_') not in poses: not_tracked.append(f)
                
    #     print('To track: ', not_tracked)
    #     print(len(not_tracked), ' files yet to track')

    #     store = "Utilities/file_io/files_to_track.yml"
    #     with open(store, 'w') as out:
    #         yaml.dump([os.path.split(u)[-1]+".mp4" for u in not_tracked], out)


if __name__ == "__main__":
    sort_mantis_files()
    convert_tdms_to_mp4()
    # print(get_list_uncoverted_tdms_videos())