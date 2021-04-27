from pathlib import Path

MAIN_FLD = Path('D:\\Dropbox (UCL)\\Rotation_vte\\Writings\\BehavPaper\\figures')

import sys
sys.path.append('./')

from pyinspect import install_traceback

install_traceback()



from loguru import logger
from rich.logging import RichHandler

logger.configure(
    handlers=[{"sink": RichHandler(markup=True), "format": "{message}"}]
)