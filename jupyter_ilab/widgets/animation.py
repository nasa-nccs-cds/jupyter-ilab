import matplotlib.widgets
import matplotlib.patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from  matplotlib.transforms import Bbox
from matplotlib.colors import LinearSegmentedColormap, Normalize, ListedColormap
import matplotlib.pyplot as plt
from matplotlib.dates import num2date
from sklearn.linear_model import LinearRegression
from threading import  Thread
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
import matplotlib as mpl
import pandas as pd
import xarray as xa
import numpy as np
from typing import List, Union, Dict, Callable, Tuple, Optional
import time, math, atexit, json
from enum import Enum

def get_color_bounds( color_values: List[float] ) -> List[float]:
    color_bounds = []
    for iC, cval in enumerate( color_values ):
        if iC == 0: color_bounds.append( cval - 0.5 )
        else: color_bounds.append( (cval + color_values[iC-1])/2.0 )
    color_bounds.append( color_values[-1] + 0.5 )
    return color_bounds

class ADirection(Enum):
    BACKWARD = -1
    STOP = 0
    FORWARD = 1

class EventSource(Thread):

    def __init__( self, action: Callable, **kwargs ):
        Thread.__init__(self)
        self.event = None
        self.action = action
        self.interval = kwargs.get( "delay",0.01 )
        self.active = False
        self.running = True
        self.daemon = True
        atexit.register( self.exit )

    def run(self):
        while self.running:
            time.sleep( self.interval )
            if self.active:
                plt.pause( 0.05 )
                self.action( self.event )

    def activate(self, delay = None ):
        if delay is not None: self.interval = delay
        self.active = True

    def deactivate(self):
        self.active = False

    def exit(self):
        self.running = False

class PageSlider(matplotlib.widgets.Slider):

    def __init__(self, ax: Axes, numpages = 10, valinit=0, valfmt='%1d', **kwargs ):
        self.facecolor=kwargs.get('facecolor',"yellow")
        self.activecolor = kwargs.pop('activecolor',"blue" )
        self.stepcolor = kwargs.pop('stepcolor', "#ff6f6f" )
        self.animcolor = kwargs.pop('animcolor', "#6fff6f" )
        self.on_animcolor = kwargs.pop('on-animcolor', "#006622")
        self.fontsize = kwargs.pop('fontsize', 10)
        self.animation_controls = kwargs.pop('dynamic', True )
        self.maxIndexedPages = 24
        self.numpages = numpages
        self.init_anim_delay: float = 0.5   # time between timer events in seconds
        self.anim_delay: float = self.init_anim_delay
        self.anim_delay_multiplier = 1.5
        self.anim_state = ADirection.STOP
        self.axes = ax
        self.event_source = EventSource( self.step, delay = self.init_anim_delay )

        super(PageSlider, self).__init__(ax, "", 0, numpages, valinit=valinit, valfmt=valfmt, **kwargs)

        self.poly.set_visible(False)
        self.vline.set_visible(False)
        self.pageRects = []
        indexMod = math.ceil( self.numpages / self.maxIndexedPages )
        for i in range(numpages):
            facecolor = self.activecolor if i==valinit else self.facecolor
            r  = matplotlib.patches.Rectangle((float(i)/numpages, 0), 1./numpages, 1, transform=ax.transAxes, facecolor=facecolor)
            ax.add_artist(r)
            self.pageRects.append(r)
            if i % indexMod == 0:
                ax.text(float(i)/numpages+0.5/numpages, 0.5, str(i+1), ha="center", va="center", transform=ax.transAxes, fontsize=self.fontsize)
        self.valtext.set_visible(False)

        divider = make_axes_locatable(ax)
        bax = divider.append_axes("right", size="5%", pad=0.05)
        fax = divider.append_axes("right", size="5%", pad=0.05)
        self.button_back = matplotlib.widgets.Button(bax, label='$\u25C1$', color=self.stepcolor, hovercolor=self.activecolor)
        self.button_forward = matplotlib.widgets.Button(fax, label='$\u25B7$', color=self.stepcolor, hovercolor=self.activecolor)
        self.button_back.label.set_fontsize(self.fontsize)
        self.button_forward.label.set_fontsize(self.fontsize)
        self.button_back.on_clicked(self.step_backward)
        self.button_forward.on_clicked(self.step_forward)

        if self.animation_controls:
            afax = divider.append_axes("left", size="5%", pad=0.05)
            asax = divider.append_axes("left", size="5%", pad=0.05)
            abax = divider.append_axes("left", size="5%", pad=0.05)
            self.button_aback    = matplotlib.widgets.Button( abax, label='$\u25C0$', color=self.animcolor, hovercolor=self.activecolor)
            self.button_astop = matplotlib.widgets.Button( asax, label='$\u25FE$', color=self.animcolor, hovercolor=self.activecolor)
            self.button_aforward = matplotlib.widgets.Button( afax, label='$\u25B6$', color=self.animcolor, hovercolor=self.activecolor)

            self.button_aback.label.set_fontsize(self.fontsize)
            self.button_astop.label.set_fontsize(self.fontsize)
            self.button_aforward.label.set_fontsize(self.fontsize)
            self.button_aback.on_clicked(self.anim_backward)
            self.button_astop.on_clicked(self.anim_stop)
            self.button_aforward.on_clicked(self.anim_forward)

    def reset_buttons(self):
        if self.animation_controls:
            for button in [ self.button_aback, self.button_astop, self.button_aforward ]:
                button.color = self.animcolor
            self.refesh()

    def refesh(self):
        self.axes.figure.canvas.draw()

    def start(self):
        self.event_source.start()

    def _update(self, event):
        super(PageSlider, self)._update(event)
        i = int(self.val)
        if i >=self.valmax: return
        self._colorize(i)

    def _colorize(self, i):
        for j in range(self.numpages):
            self.pageRects[j].set_facecolor(self.facecolor)
        self.pageRects[i].set_facecolor(self.activecolor)

    def step( self, event=None ):
        if   self.anim_state == ADirection.FORWARD:  self.forward(event)
        elif self.anim_state == ADirection.BACKWARD: self.backward(event)

    def forward(self, event=None):
        current_i = int(self.val)
        i = current_i+1
        if i >= self.valmax: i = self.valmin
        self.set_val(i)
        self._colorize(i)

    def backward(self, event=None):
        current_i = int(self.val)
        i = current_i-1
        if i < self.valmin: i = self.valmax -1
        self.set_val(i)
        self._colorize(i)

    def step_forward(self, event=None):
        self.anim_stop()
        self.forward(event)

    def step_backward(self, event=None):
        self.anim_stop()
        self.backward(event)

    def anim_forward(self, event=None):
        if self.anim_state == ADirection.FORWARD:
            self.anim_delay = self.anim_delay / self.anim_delay_multiplier
            self.event_source.interval = self.anim_delay
        elif self.anim_state == ADirection.BACKWARD:
            self.anim_delay = self.anim_delay * self.anim_delay_multiplier
            self.event_source.interval = self.anim_delay
        else:
            self.anim_delay = self.init_anim_delay
            self.anim_state = ADirection.FORWARD
            self.event_source.activate( self.anim_delay )
            self.button_aforward.color = self.on_animcolor
            self.refesh()

    def anim_backward(self, event=None):
        if self.anim_state == ADirection.FORWARD:
            self.anim_delay = self.anim_delay * self.anim_delay_multiplier
            self.event_source.interval = self.anim_delay
        elif self.anim_state == ADirection.BACKWARD:
            self.anim_delay = self.anim_delay / self.anim_delay_multiplier
            self.event_source.interval = self.anim_delay
        else:
            self.anim_delay = self.init_anim_delay
            self.anim_state = ADirection.BACKWARD
            self.event_source.activate( self.anim_delay )
            self.button_aback.color = self.on_animcolor
            self.refesh()

    def anim_stop(self, event=None):
        if self.anim_state != ADirection.STOP:
            self.anim_delay = self.init_anim_delay
            self.anim_state = ADirection.STOP
            self.event_source.deactivate()
            self.reset_buttons()

class SliceAnimation:

    def __init__(self, data_arrays: Union[xa.DataArray,List[xa.DataArray]], **kwargs ):
        self.frames: np.ndarray = None
        self.plot_grid = None
        self.metrics_scale =  kwargs.get( 'metrics_scale', None )
        self.data: List[xa.DataArray] = self.preprocess_inputs(data_arrays)
        self.global_mean = None
        self.global_bounds: Bbox = None
        self.global_crange = None
        self.metrics_range = None
        self.name = self.get_name( self.data[0] )
        self.plot_axes = None
        self.metrics_alpha = kwargs.get( "metrics_alpha", 0.7 )
        self.metrics_plots = {}
        self.figure: Figure = plt.figure()
        self.plot_grid = self.figure.add_gridspec( 3, 2 )
        self.images: Dict[int,AxesImage] = {}
        self.nPlots = min( len(self.data), 4 )
        self.metrics: Dict = dict( blue="ts" ) # kwargs.get("metrics", {})
        self.frame_marker: Line2D = None
        self.setup_plots(**kwargs)
        self.dataLims = {}
        self.z_axis = kwargs.pop('z', 0)
        self.z_axis_name = self.data[0].dims[ self.z_axis ]
        self.x_axis = kwargs.pop( 'x', 2 )
        self.x_axis_name = self.data[0].dims[ self.x_axis ]
        self.y_axis = kwargs.pop( 'y', 1 )
        self.y_axis_name = self.data[0].dims[ self.y_axis ]
        self.currentFrame = 0
        self.currentPlot = 0

        self.add_plots( **kwargs )
        self.add_slider( **kwargs )
        self._update(0)

    @property
    def toolbarMode(self) -> str:
        return self.figure.canvas.toolbar.mode

    def preprocess_inputs(self, data_arrays: Union[xa.DataArray,List[xa.DataArray]]  ) -> List[xa.DataArray]:
        inputs_list = data_arrays if isinstance(data_arrays, list) else [data_arrays]
        axis0_lens = [ input.shape[0] for input in inputs_list ]
        self.nFrames = max(axis0_lens)
        target_input = inputs_list[ axis0_lens.index( self.nFrames ) ]
        self.frames = target_input.coords[ target_input.dims[0] ].values
        return inputs_list

    def check_axes( cls, inputs: List[xa.DataArray] ):
        coord_lists = [ [], [], [] ]
        for input in inputs:
            for iC in range(3):
                coord_lists[iC].append( input.coords[ input.dims[iC]] )
        for clist in coord_lists: cls.test_coord_equivalence( clist )

    def test_coord_equivalence( cls, coords: List[xa.DataArray] ):
        c0 = coords[0]
        for c1 in coords[1:]:
            if (c0.shape[0] != c1.shape[0]) or ( c0.values[0] != c1.values[0] ) or ( c0.values[1] != c1.values[1] ):
                raise Exception( "Coordinate Mismatch")

    @classmethod
    def time_merge( cls, data_arrays: List[xa.DataArray], **kwargs ) -> xa.DataArray:
        time_axis = kwargs.get('time',None)
        frame_indices = range( len(data_arrays) )
        merge_coord = pd.Index( frame_indices, name=kwargs.get("dim","time") ) if time_axis is None else time_axis
        result: xa.DataArray =  xa.concat( data_arrays, dim=merge_coord )
        return result

    def setup_plots( self, **kwargs ):
        self.plot_grid = self.figure.add_gridspec(3,2)
        if self.nPlots == 1:    gsl = [ self.plot_grid[:-1, :] ]
        elif self.nPlots == 2:  gsl = [ self.plot_grid[:-1, 0], self.plot_grid[:-1, 1]  ]
        elif self.nPlots == 3:  gsl = [ self.plot_grid[0, 0], self.plot_grid[:-1, 1], self.plot_grid[0, 1], self.plot_grid[1, 0] ]
        elif self.nPlots == 4:  gsl = [ self.plot_grid[0, 0], self.plot_grid[:-1, 1], self.plot_grid[0, 1], self.plot_grid[1, 0], self.plot_grid[1, 1] ]
        else: raise Exception( f"Unsupported number of plots: {self.nPlots}" )
        self.metrics_plot = self.figure.add_subplot( self.plot_grid[2, :] )
        self.plot_axes = np.array( [ self.figure.add_subplot(gs) for gs in gsl ] )
        self.figure.tight_layout()
        self.slider_axes: Axes = self.figure.add_axes([0.1, 0.01, 0.8, 0.04])  # [left, bottom, width, height]

    def invert_yaxis(self):
        self.plot_axes[0].invert_yaxis()

    def get_xy_coords(self, iPlot: int ) -> Tuple[ np.ndarray, np.ndarray ]:
        return self.get_coord( iPlot, self.x_axis ), self.get_coord( iPlot, self.y_axis )

    def get_anim_coord(self, iPlot: int ) -> np.ndarray:
        return self.get_coord( iPlot, 0 )

    def get_coord(self, iPlot: int, iCoord: int ) -> np.ndarray:
        data = self.data[iPlot]
        return data.coords[ data.dims[iCoord] ].values

    def create_cmap( self, cmap_spec: Union[str,Dict] ):
        if isinstance(cmap_spec,str):
            cmap_spec =  json.loads(cmap_spec)
        range = cmap_spec.pop("range",None)
        colors = cmap_spec.pop("colors",None)
        if colors is None:
            cmap = cmap_spec.pop("cmap","jet")
            norm = Normalize(*range) if range else None
            return dict( cmap=cmap, norm=norm, cbar_kwargs=dict(cmap=cmap, norm=norm, orientation='vertical'), tick_labels=None )
        else:
            if isinstance( colors, np.ndarray ):
                return dict( cmap=LinearSegmentedColormap.from_list('my_colormap', colors) )
            rgbs = [ cval[2] for cval in colors ]
            cmap: ListedColormap = ListedColormap( rgbs )
            tick_labels = [ cval[1] for cval in colors ]
            color_values = [ float(cval[0]) for cval in colors]
            color_bounds = get_color_bounds(color_values)
            norm = mpl.colors.BoundaryNorm( color_bounds, len( colors )  )
            cbar_args = dict( cmap=cmap, norm=norm, boundaries=color_bounds, ticks=color_values, spacing='proportional',  orientation='vertical')
            return dict( cmap=cmap, norm=norm, cbar_kwargs=cbar_args, tick_labels=tick_labels )

    def update_metrics_marker(self):
        data_array: xa.DataArray = self.data[self.currentPlot]
        tcoord = data_array.coords[data_array.dims[0]].values[self.currentFrame]
        axis: Axes = self.metrics_plot
        x = [ tcoord, tcoord ]
        y = [ axis.dataLim.y0, axis.dataLim.y1 ] if self.metrics_range is None else self.metrics_range
        if self.frame_marker == None:
            self.frame_marker, = axis.plot( x, y, color="green", lw=3, alpha=0.5 )
        else:
            self.frame_marker.set_data( x, y )

    def create_image(self, iPlot: int, **kwargs ) -> AxesImage:
        data: xa.DataArray = self.data[iPlot]
        subplot: Axes = self.plot_axes[iPlot]
        cm = self.create_cmap( data.attrs.get("cmap",{}) )
        z: xa.DataArray =  data[ 0, :, : ]   # .transpose()
        color_tick_labels = cm.pop( 'tick_labels', None )
        image: AxesImage = z.plot.imshow( ax=subplot, **cm )
        self._cidpress = image.figure.canvas.mpl_connect('button_press_event', self.onMouseClick)
        self._cidrelease = image.figure.canvas.mpl_connect('button_release_event', self.onMouseRelease )
        subplot.callbacks.connect('ylim_changed', self.on_lims_change)
        if color_tick_labels is not None: image.colorbar.ax.set_xticklabels( color_tick_labels )
        subplot.title.set_text( data.name )
        overlays = kwargs.get( "overlays", {} )
        for color, overlay in overlays.items():
            overlay.plot( ax=subplot, color=color, linewidth=2 )
        return image

    def on_lims_change(self, ax ):
        for iPlot in range(len(self.plot_axes)):
             if ax == self.plot_axes[iPlot]:
                 (x0, x1) = ax.get_xlim()
                 (y0, y1) = ax.get_ylim()
                 print(f"ZOOM Event: Updated bounds: ({x0},{x1}), ({y0},{y1})")
                 crange = self.update_metrics_plot( Bbox.from_bounds( x0, y0, x1-x0, y1-y0 ) )
                 if crange:
                    self.images[iPlot].set_clim( crange )
                    self.update_metrics_marker()

    def get_name(self, var: xa.DataArray) -> str:
        vname = var.attrs.get( "long_name", var.attrs.get( "standard_name", var.name) )
        units = var.attrs.get( "units", None )
        return f"{vname} ({units})" if units else vname

    def compute_analytics( self, bounds: Bbox = None ) -> Tuple[ xa.DataArray, xa.DataArray, List, str, Optional[List] ]:
        data_array: xa.DataArray = self.data[self.currentPlot]
        axes = self.plot_axes[self.currentPlot]
        tcoord, ycoord, xcoord = data_array.coords[data_array.dims[0]], data_array.coords[data_array.dims[1]], data_array.coords[data_array.dims[2]]
        crange = None

        if bounds is None or np.allclose( self.global_bounds.extents, bounds.extents, 0.0, 0.01 ):
            if self.global_mean   is None: self.global_mean   = data_array.mean( dim=data_array.dims[1:] )
            if self.global_bounds is None: self.global_bounds = axes.dataLim
            if self.global_crange is None:
                image0 = data_array[0,:,:]
                self.global_crange = [ image0.min(), image0.max()]
            tsdata = self.global_mean
            crange = self.global_crange
            title = f" Global Spatial Average"
        elif bounds.width == 0 or bounds.height == 0:
            selected_point_args = { data_array.dims[1]: bounds.y0, data_array.dims[2]: bounds.x0 }
            print( f"METRICS: {selected_point_args}")
            tsdata =  data_array.sel( **selected_point_args, method='nearest' )
            title = f"Values at Location ({bounds.x0:.2f} {bounds.y0:.2f})"
        else:
            sbounds = { data_array.dims[2]: slice(bounds.x0,bounds.x1), data_array.dims[1]: slice(bounds.y0,bounds.y1) }
            subset = data_array.sel( **sbounds )
            tsdata = subset.mean( dim=data_array.dims[1:] )
            visible_subset = subset[self.currentFrame,:,:]
            crange = [ visible_subset.min(), visible_subset.max() ]
            title = f"Spatial Average over Domain ({bounds.x0:.2f},{bounds.x1:.2f}), ({bounds.y0:.2f},{bounds.y1:.2f})"

        regressor = LinearRegression()
        tvar = np.array(list(range(tcoord.size))).reshape([tcoord.size, 1])
        T = regressor.fit( tvar, tsdata.values )
        trend = [T.intercept_, T.intercept_ + T.coef_[0] * tcoord.size]
        return tsdata, tcoord, trend, title, crange

    def update_metrics_plot( self, bounds: Bbox = None ):
        axis = self.metrics_plot
        color = "blue"
        values, tcoord, trend, title, crange = self.compute_analytics( bounds )
        axis.title.set_text( title )
        t = tcoord.values
        mplot = self.metrics_plots.get( color, None )
        if mplot is None:
            line, = axis.plot( t, values, color=color, alpha=self.metrics_alpha )
            line.set_label(values.name)
            self.metrics_plots[ color ] = line
            self.trend_line, = axis.plot( [t[0],t[-1]], trend, color="red" )
        else:
            mplot.set_data( t, values )
            self.trend_line.set_data( [t[0],t[-1]], trend )
            self.metrics_range = [ values.min(), values.max() ]
            axis.set_ylim( *self.metrics_range )
            axis.set_yticks( np.linspace( *self.metrics_range, 7 ) )
            plt.draw()
        return crange

    def create_metrics_plot(self):
        axis = self.metrics_plot
        axis.title.set_text("Metrics")
        if self.metrics_scale is not None:
            axis.set_yscale( self.metrics_scale )
        markers = self.metrics.pop('markers',{})
        self.update_metrics_plot()
        for color, value in markers.items():
            x = [value, value]
            y = [axis.dataLim.y0, axis.dataLim.y1]
            line, = axis.plot(x, y, color=color)
        axis.legend()

    def update_plots(self ):
        tval = self.frames[ self.currentFrame ]
        for iPlot in range(self.nPlots):
            subplot: Axes = self.plot_axes[iPlot]
            data = self.data[iPlot]
            frame_image = data.isel( **{ data.dims[0]: self.currentFrame } )
            try:                tval1 = frame_image.time.values
            except Exception:   tval1 = tval
            self.images[iPlot].set_data( frame_image )
            stval = str(tval1).split("T")[0]
            subplot.title.set_text( f"{self.name} [{self.currentFrame}]: {stval}" )
        self.update_metrics_marker()

    def onMouseRelease(self, event):
        pass

    def onMouseClick(self, event):
        if event.xdata != None and event.ydata != None and not self.toolbarMode:
            axis: Axes = self.metrics_plot
            if event.inaxes == axis:
                data_array: xa.DataArray = self.data[self.currentPlot]
                taxis = data_array.coords[data_array.dims[0]].values
                dtime = np.datetime64( num2date( event.xdata ) )
                idx = np.searchsorted( taxis, dtime, side="left")
                print( f"onMetricsClick: {event.xdata} -> {dtime} --> {idx}" )
                self.slider.set_val( idx )
            for iPlot in range( len(self.plot_axes) ):
                if event.inaxes ==  self.plot_axes[iPlot]:
                    self.currentPlot = iPlot
                    print(f"onImageClick: {event.xdata} {event.ydata}")
                    self.update_metrics_plot( Bbox.from_bounds( event.xdata, event.ydata, 0, 0 ) )
                    self.dataLims[iPlot] = event.inaxes.dataLim

    def datalims_changed(self, iPlot: int ) -> bool:
        previous_datalims: Bbox = self.dataLims[iPlot]
        new_datalims: Bbox = self.plot_axes[iPlot].dataLim
        return previous_datalims.bounds != new_datalims.bounds

    def add_plots(self, **kwargs ):
        for iPlot in range(self.nPlots):
            self.images[iPlot] = self.create_image( iPlot, **kwargs )
        self.create_metrics_plot()

    def add_slider(self,  **kwargs ):
        self.slider = PageSlider( self.slider_axes, self.nFrames )
        self.slider_cid = self.slider.on_changed(self._update)

    def _update( self, val ):
        tval = self.slider.val
        self.currentFrame = int( tval )
        self.update_plots()

    def show(self):
        self.slider.start()
        plt.show()

    def start(self):
        self.slider.start()

if __name__ == '__main__':
    from jupyter_ilab.util.cip import CIP

    data_array: xa.DataArray = CIP.data_array( "merra2", "tas" )
    animator = SliceAnimation( data_array )
    animator.show()
