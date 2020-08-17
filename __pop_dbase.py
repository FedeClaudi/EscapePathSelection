import sys
sys.path.append("../")

import pyexcel

from paper import paths
from paper.dbase.TablesDefinitionsV4 import *
from paper.dbase.utils import insert_entry_in_table




def populate_mice_table():
        """ Populates the Mice() table from the database"""
        table = Mouse
        loaded_excel = pyexcel.get_records(file_name=paths.mice_records)

        for m in loaded_excel:
            if not m['']: continue

            mouse_data = dict(
                mouse_id = m[''],
                strain = m['Strain'],
                sex = 'M',
            )
            insert_entry_in_table(mouse_data['mouse_id'], 'mouse_id', mouse_data, table)

def populate_sessions_table():
    """  Populates the sessions table """
    mice = Mouse().fetch(as_dict=True)
    micenames = list(pd.DataFrame(mice).mouse_id.values)
    loaded_excel = pyexcel.get_records(file_name=paths.exp_records)

    for session in loaded_excel:
        # # Get mouse name
        mouse_id = session['MouseID']
        for mouse in micenames:
            if mouse_id == mouse: 
                break
            else:
                original_mouse = mouse
                mouse = mouse.replace('_', '')
                mouse = mouse.replace('.', '')
                if mouse == mouse_id:
                    mouse_id = original_mouse
                    break

        # Get session name
        session_name = '{}_{}'.format(session['Date'], session['MouseID'])
        session_date = '20'+str(session['Date'])

        # Get experiment name
        experiment_name = session['Experiment']

        # Insert into table
        session_data = dict(
            uid = str(session['Sess.ID']), 
            session_name=session_name,
            mouse_id=mouse_id,
            date=session_date,
            experiment_name = experiment_name
        )

        if not session['Experiment']: continue
        try:
            insert_entry_in_table(session_data['session_name'], 'session_name', session_data, Session)
        except:
            a = 1

        # Insert into metadata part table
        part_dat = dict(
            session_name=session_data["session_name"],
            uid=session_data["uid"],
            maze_type= int(session["Maze type"]),
            naive = int(session["Naive"]),
            lights = int(session["Lights"]),
            mouse_id=mouse_id,
        )

        insert_entry_in_table(part_dat['session_name'], 'session_name', part_dat, Session.Metadata)

        # Insert into shelter metadata
        part_dat = dict(
            session_name=session_data["session_name"],
            uid=session_data["uid"],
            shelter= int(session["Shelter"]),
            mouse_id=mouse_id,

        )

        insert_entry_in_table(part_dat['session_name'], 'session_name', part_dat, Session.Shelter)



if __name__ == "__main__":
    # populate_mice_table()
    # populate_sessions_table()

    # MazeComponents.populate(display_progress=True)
    Recording.populate(display_progress=True)
    CCM.populate(display_progress=True)


    # Trials.populate(display_progress=True, suppress_errors=True)