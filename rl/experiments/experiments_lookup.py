from rl.environment.mazes import (
    M0,
    M1,
    M2,
    M3,
    PsychometricM1,
    PsychometricM6,
    MB,
    PsychometricM2,
    Open,
    Wall,
)

from models import (
    QTableModel,
    QTableTracking,
    FeudalQLearningTable,
    SuccessorOptions,
    DynaQModel,
    DynaQTracking,
    MAXQModel,
    InfluenceZones,
    InfluenceZonesTracking
)

EXPERIMENTS = dict(
    # SIMULATED: fake mazes
    Open = dict(
        maze = Open,
        agent = QTableModel,
        blocked= False,
        shortcut=False,
        repeats=1,
    ),

    Wall = dict(
        maze = Wall,
        agent = DynaQModel,
        blocked= False,
        shortcut=False,
        repeats=1,
    ),

    M0 = dict(
        maze = M0,
        agent = QTableModel,
        blocked= False,
        shortcut=False,
        repeats=1,
    ),

    M1 = dict(
        maze = M1,
        agent = DynaQModel,
        blocked= False,
        shortcut=False,
        repeats=1,
    ),

    M2 = dict(
        maze = M2,
        agent = DynaQModel,
        blocked= True,
        shortcut=False,
        repeats=1,
    ),

    M3 = dict(
        maze = M3,
        agent = DynaQModel,
        blocked= False,
        shortcut=True,
        repeats=1,
    ),

    # REAL: psychometric mazes and real tracking data
    PsychometricM1 = dict(
        maze = PsychometricM1,
        agent = InfluenceZonesTracking,
        blocked = False,
        shortcut = False,
        repeats = 1,
        exploration_file_name='M1',
        trial_number = 37,
    ),

    PsychometricM2 = dict(
        maze = PsychometricM2,
        agent = InfluenceZonesTracking,
        blocked = False,
        shortcut = False,
        repeats = 1,
        exploration_file_name='M2',
        trial_number = 2,
    ),

    PsychometricM6 = dict(
        maze = PsychometricM6,
        agent = InfluenceZonesTracking,
        blocked = False,
        shortcut = False,
        repeats = 1,
        exploration_file_name='M6',
        trial_number = 15,
    ),

    MB = dict(
        maze = MB,
        agent = DynaQTracking,
        blocked = True,
        shortcut = False,
        repeats = 1,
        exploration_file_name='MB',
        trial_number = 6,
    ),

    # REAL mazes but with simulated agents
    PsychometricM1_SIM = dict(
        maze = PsychometricM1,
        agent = InfluenceZones,
        blocked = False,
        shortcut = False,
        repeats = 1,
    ),

    PsychometricM2_SIM = dict(
        maze = PsychometricM2,
        agent = SuccessorOptions,
        blocked = False,
        shortcut = False,
        repeats = 1,
    ),    

    PsychometricM6_SIM = dict(
        maze = PsychometricM6,
        agent = QTableModel,
        blocked = False,
        shortcut = False,
        repeats = 1,
    ),    

    MB_SIM = dict(
        maze = MB,
        agent = DynaQModel,
        blocked = True,
        shortcut = False,
        repeats = 1,
    ),   
)