import os
import sys
# from dotenv import load_dotenv
import inspect

# load_dotenv()  # take environment variables from .env.

try:
    sys.path.append(os.path.split(os.environ["PLANETARY_DATA_PATH"])[0])
except KeyError:
    pass

src_dir = os.path.dirname(os.path.abspath(inspect.getsourcefile(lambda: 0)))

os.environ["FRAG_SRC"] = src_dir

from .centaur_dict import *
from .orbit_conversions import *
from .tle_functions import *
from .unscented_transform import *
from .vimpel_case import *
from .vimpel_functions import *