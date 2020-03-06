from jupyter_ilab.util.cip import CIP
import xarray as xa
import matplotlib.pyplot as plt

def on_press( event ):
    print( "on_press")

def on_release( event ):
    print( "on_release")

data_array: xa.DataArray = CIP.data_array( "merra2", "tas" )
image_data = data_array[0,:,:].squeeze()
image =  image_data.plot.imshow( cmap="jet")
cidpress = image.figure.canvas.mpl_connect('button_press_event', on_press)
cidrelease = image.figure.canvas.mpl_connect('button_release_event', on_release)

plt.show()