from rl.models.qtable import QTableModel, QTableTracking
from rl.models.dynaQ import DynaQModel, DynaQTracking
from rl.models.influence_zones import InfluenceZones, InfluenceZonesTracking

from loguru import logger
from rich.logging import RichHandler

logger.configure(handlers=[{"sink":RichHandler(markup=True), "format":"{message}"}])
