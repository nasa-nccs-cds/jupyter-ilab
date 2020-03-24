from jupyter_ilab.util.cip import CIP
from typing import List, Union, Dict, Callable, Tuple, Optional
from jupyter_ilab.widgets.animation import SliceAnimation
import xarray as xa
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]

vars = [ "tas", "huss" ]
data_arrays: List[xa.DataArray] = [ CIP.data_array( "merra2", var ) for var in vars ]

animator = SliceAnimation( data_arrays )
animator.start()


