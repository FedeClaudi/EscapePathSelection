import sys
sys.path.append("../")

from paper.dbase.TablesDefinitionsV4 import *

sessions = (Session & "experiment_name='shortcut'").fetch('session_name')

TrackingData.populate(display_progress=True)

# tables = [TrackingData.BodyPartData,]
# for table in tables:
#     for sess in sessions:
#         print(f'Deleting {sess} from {table}')
#         # print((table &  f"session_name='{sess}'"))
#         try:
#             # for bp in ['body']:  & f'bpname="{bp}"'
#             (table & f"session_name='{sess}'").delete_quick()
#         except  Exception as e:
#             print(f'Didnt delete {sess} from {table} because of {e}')
