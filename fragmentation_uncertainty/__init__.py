import os
import sys
# from dotenv import load_dotenv
import inspect

# load_dotenv()  # take environment variables from .env.
# sys.path.append(os.path.abspath('../planetary_data'))



try:
    sys.path.append(os.path.split(os.environ["PLANETARY_DATA_PATH"])[0])
except KeyError:
    pass

src_dir = os.path.dirname(os.path.abspath(inspect.getsourcefile(lambda: 0)))

os.environ["FRAG_SRC"] = src_dir

from .catalog import *
from .centaur_dict import *
from .ode import *
from .orbit_conversions import *
from .perturbation_functions import *
from .propagation import *
from .unscented_transform import *
from .vimpel_case import *
from .vimpel_functions import *