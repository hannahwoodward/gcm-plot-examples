from dask.diagnostics import ProgressBar
from datetime import timedelta
from pathlib import Path

import numpy as np
import xarray

# https://docs.xarray.dev/en/stable/internals/extending-xarray.html
@xarray.register_dataset_accessor('gcm_data')
class GcmDataAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def save_progressive(
        self,
        filepath,
        engine='netcdf4',
        unlimited_dims=['time']
    ):
        write = self._obj.to_netcdf(
            filepath,
            compute=False,
            engine=engine,
            unlimited_dims=unlimited_dims
        )
        
        with ProgressBar():
            write.compute()

    def slice_orbits(self, start, end):
        data = self._obj.copy()
        
        time_start = data.time.values[0] + timedelta(days=start)
        time_end = time_start + timedelta(days=(end - start))
        data_slice = data.sel(time=slice(time_start, time_end))

        return data_slice

    def slice_orbits_last(self, x):
        data = self._obj.copy()
        
        time_end = data.time.values[-1]
        time_start = time_end - timedelta(days=x)
        data_sliced = data.sel(time=slice(time_start, time_end))

        return data_sliced

    def regrid_data(
        self,
        grid,
        filename=None,
        regrid_kwargs={}
    ):
        data = self._obj.copy()
    
        regrid = xesmf.Regridder(
            data,
            grid,
            **regrid_kwargs
        )
    
        data_regridded = regrid(data)
    
        # filename != None and data_regridded.to_netcdf(filename)
        filename != None and data_regridded._obj.save_progressive(filename)
    
        return data_regridded


    def weighted_lat(
        self,
        dim='lat'
    ):
        data = self._obj.copy()
        
        weights = np.cos(np.deg2rad(data[dim]))
        weights.name = 'weights'
        
        return data.weighted(weights)