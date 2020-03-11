import xarray as xa
from jupyter_ilab.widgets.animation import SliceAnimation
from typing import List, Union, Dict, Callable, Tuple, Optional
import time, math, atexit, json
from jupyter_ilab.util.cip import CIP

input_file = "/Users/tpmaxwel/Dropbox/Tom/Data/Aviris/ang20170714t213741_rfl_v2p9/ang20170714t213741_corr_v2p9_img"
data_array: xa.DataArray = xa.open_rasterio(input_file)
animator = SliceAnimation(data_array)
animator.show()
