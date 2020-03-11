from glob import glob
import xarray as xa
import numpy as np
import numpy.ma as ma
import rioxarray
import os
import logging, unittest

ref_nodata = -9999

class FormatConverter:

    def __init__( self, reference_grid: str = None ):
        self.ref_grid: xa.DataArray = None
        if reference_grid is not None:
            self.reference_grid = self.open( reference_grid )
            self.reference_grid = self.reference_grid.where( self.reference_grid != ref_nodata, float("nan"))
            self.ref_median = ma.median( ma.masked_invalid(self.reference_grid) )

    def convert(self, files_glob: str, varname: str, **kwargs ):
        for file in glob(files_glob):
            input: xa.DataArray = self.open( file, varname, **kwargs )
            self.write( os.path.splitext(file)[0], input, **kwargs )

    def open(self, file_path: str, varname: str = None, **kwargs ) -> xa.DataArray:
        print( f"Opening input file {file_path}")

        if os.path.splitext(file_path)[1] in ["nc","nc4"]:
            result = xa.open_dataset( file_path )[varname].squeeze()
        else:
            result = xa.open_rasterio( file_path ).squeeze()
        return result

    def write( self, base_file_name: str, data_array: xa.DataArray, **kwargs ):
        output_file =  base_file_name + ".tif"
        print(f"Writing output_file {output_file}")
        invert_y = kwargs.get( 'invert_y', False )
        output_nodata = kwargs.get('output_nodata', None )
        if self.ref_grid is not None:
            data_array: xa.DataArray = data_array.assign_coords( self.ref_grid.coords )
            data_array.attrs['crs'] = self.ref_grid.crs
            data_array.attrs['transform'] = self.ref_grid.transform
        if invert_y:
            ydim = data_array.dims[0]
            data_array.coords[ydim] = data_array.coords[ydim][::-1]
        da_median = ma.median( ma.masked_invalid(data_array) )
        data_array = data_array * (self.ref_median/da_median)
        if output_nodata is not None:
            data_array = data_array.fillna( output_nodata )
        data_array.rio.to_raster(output_file)


if __name__ == '__main__':
    DATA_DIR = '/Users/tpmaxwel/Dropbox/Tom/InnovationLab/results/Aviris/constructed_images'
    input_files = f'{DATA_DIR}/*.nc'
    reference_grid: str = f'{DATA_DIR}/ang20170714t213741_Avg-Chl.tif'
    fc = FormatConverter( reference_grid )
    fc.convert( input_files, 'constructed_image', invert_y = True, output_nodata = ref_nodata )