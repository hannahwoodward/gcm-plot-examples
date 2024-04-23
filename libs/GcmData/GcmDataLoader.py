from dask.diagnostics import ProgressBar
from datetime import timedelta
import numpy as np
import xarray

from . import GcmData


class GcmDataLoader():
    def __init__(
        self,
        id,
        path='',
        keep_vars=[]
    ):
        self.id = id

        self.path = str(path).format(id=id)
        self.keep_vars = keep_vars

    def load(self):
        data = xarray.open_mfdataset(
            self.path,
            combine='nested',
            concat_dim='time',
            autoclose=True,
            use_cftime=True
        )

        return data

    def standardise_vars(self, data_orig):
        data = data_orig.copy()

        var_map = self.var_map()
        var_attrs = self.var_attrs()
        for v in var_map:
            mapping = var_map[v]
            if type(mapping) == str:
                mapping = lambda x: x[var_map[v]]

            data = data.assign({ v: mapping })
            if v in var_attrs:
                data[v].attrs = {}
                data[v] = data[v].assign_attrs({
                    **var_attrs[v]
                })

        # Only keep standardised variables
        keep_vars = [
            *self.keep_vars,
            *var_map.keys()
        ]
        data = data[keep_vars]

        return data

    def var_attrs(self, v=None):
        '''
        Attributes of mapped variables.
        
        CMIP6 variable information taken from
        https://clipc-services.ceda.ac.uk/dreq/mipVars.html
        '''
        all_attrs = {
            'areacella': {
                'variable_long_name': 'Grid-Cell Area for Atmospheric Grid Variables',
                'variable_units': 'm2',
                'variable_units_latex': '$m^2$'
            },
            'clivi': {
                'variable_long_name': 'Ice Water Path',
                'variable_units': 'kg m-2',
                'variable_units_latex': '$kg$ $m^{-2}$'
            },
            'clt': {
                'variable_long_name': 'Total Cloud Cover Percentage',
                'variable_units': '%',
                'variable_units_latex': '%'
            },
            'hurs': {
                'variable_long_name': ' Near-Surface Relative Humidity',
                'variable_units': '%',
                'variable_units_latex': '%'
            },
            'huss': {
                'variable_long_name': ' Near-Surface Specific Humidity',
                'variable_units': 'kg/kg',
                'variable_units_latex': '$kg/kg$'
            },
            'evspsbl': {
                'variable_long_name': 'Evaporation Including Sublimation and Transpiration',
                'variable_units': 'kg m-2 s-1',
                'variable_units_latex': '$mm$ $s^{-1}$'
            },
            'hfls': {
                'variable_long_name': 'Surface Upward Latent Heat Flux',
                'variable_units': 'W m-2',
                'variable_units_latex': '$W$ $m^{-2}$'
            },
            'lwp': {
                'variable_long_name': 'Liquid Water Path',
                'variable_units': 'kg m-2',
                'variable_units_latex': '$kg$ $m^{-2}$'
            },
            'pr': {
                'variable_long_name': 'Precipitation',
                'variable_units': 'kg m-2 s-1',
                'variable_units_latex': '$mm$ $s^{-1}$'
            },
            'prra': {
                'variable_long_name': 'Rainfall Flux',
                'variable_units': 'kg m-2 s-1',
                'variable_units_latex': '$mm$ $s^{-1}$'
            },
            'prsn': {
                'variable_long_name': 'Snowfall Flux',
                'variable_units': 'kg m-2 s-1',
                'variable_units_latex': '$mm$ $s^{-1}$'
            },
            'ps': {
                'variable_long_name': 'Surface Air Pressure',
                'variable_units': 'Pa',
                'variable_units_latex': '$Pa$'
            },
            'rls': {
                'variable_long_name': 'Net Longwave Surface Radiation',
                'variable_units': 'W m-2',
                'variable_units_latex': '$W$ $m^{-2}$'
            }, 
            'rlds': {
                'variable_long_name': 'Surface Downwelling Longwave Radiation',
                'variable_units': 'W m-2',
                'variable_units_latex': '$W$ $m^{-2}$'
            }, 
            'rldscs': {
                'variable_long_name': 'Surface Downwelling Clear-Sky Longwave Radiation',
                'variable_units': 'W m-2',
                'variable_units_latex': '$W$ $m^{-2}$'
            },
            'rlus': {
                'variable_long_name': 'Surface Upwelling Longwave Radiation',
                'variable_units': 'W m-2',
                'variable_units_latex': '$W$ $m^{-2}$'
            },
            'rlut': {
                'variable_long_name': 'TOA Outgoing Longwave Radiation',
                'variable_units': 'W m-2',
                'variable_units_latex': '$W$ $m^{-2}$'
            }, 
            'rlutcs': {
                'variable_long_name': 'TOA Outgoing Clear-Sky Longwave Radiation',
                'variable_units': 'W m-2',
                'variable_units_latex': '$W$ $m^{-2}$'
            },
            'rss': {
                'variable_long_name': 'Net Shortwave Surface Radiation',
                'variable_units': 'W m-2',
                'variable_units_latex': '$W$ $m^{-2}$'
            },
            'rsds': {
                'variable_long_name': 'Surface Downwelling Shortwave Radiation',
                'variable_units': 'W m-2',
                'variable_units_latex': '$W$ $m^{-2}$'
            }, 
            'rsus': {
                'variable_long_name': 'Surface Upwelling Shortwave Radiation',
                'variable_units': 'W m-2',
                'variable_units_latex': '$W$ $m^{-2}$'
            }, 
            'rsdt': {
                'variable_long_name': 'TOA Incident Shortwave Radiation',
                'variable_units': 'W m-2',
                'variable_units_latex': '$W$ $m^{-2}$'
            },
            'rsut': {
                'variable_long_name': 'TOA Outgoing Shortwave Radiation',
                'variable_units': 'W m-2',
                'variable_units_latex': '$W$ $m^{-2}$'
            }, 
            'rsutcs': {
                'variable_long_name': 'TOA Outgoing Clear-Sky Shortwave Radiation',
                'variable_units': 'W m-2',
                'variable_units_latex': '$W$ $m^{-2}$'
            }, 
            'sfcWind': {
                'variable_long_name': 'Near-Surface Wind Speed',
                'variable_units': 'm s-1',
                'variable_units_latex': '$m$ $s^{-1}$'
            },
            'siconc': {
                'variable_long_name': 'Sea-Ice Area Percentage (Ocean Grid)',
                'variable_units': '%',
                'variable_units_latex': '$%$'
            },
            'sitemptop': {
                'variable_long_name': 'Surface Temperature of Sea Ice',
                'variable_units': 'K',
                'variable_units_latex': '$K$'
            },
            'sithick': {
                'variable_long_name': 'Sea Ice Thickness',
                'variable_units': 'm',
                'variable_units_latex': '$m$'
            },
            'sisnthick': {
                'note': 'May also include SND (land snow depth)',
                'variable_long_name': 'Snow Thickness',
                'variable_units': 'm',
                'variable_units_latex': '$m$'
            },
            'so': {
                'note': 'Units are dimensionless & parts per thousand',
                'variable_long_name': 'Sea Water Salinity',
                'variable_units': 'ppt',
                'variable_units_latex': '$ppt$'
            },
            'tas': {
                'variable_long_name': 'Near-Surface Air Temperature',
                'variable_units': 'K',
                'variable_units_latex': '$K$'
            },
            'tauuo': {
                'variable_long_name': 'Sea Water Surface Downward X Stress',
                'variable_units': 'N m-2',
                'variable_units_latex': '$N$ $m^{-2}$'
            },
            'tauvo': {
                'variable_long_name': 'Sea Water Surface Downward Y Stress',
                'variable_units': 'N m-2',
                'variable_units_latex': '$N$ $m^{-2}$'
            },
            'tntrl': {
                'variable_long_name': 'Tendency of Air Temperature Due to Longwave Radiative Heating',
                'variable_units': 'K s-1',
                'variable_units_latex': '$K$ $s^{-1}$'
            },
            'tntrs': {
                'variable_long_name': 'Tendency of Air Temperature Due to Shortwave Radiative Heating',
                'variable_units': 'K s-1',
                'variable_units_latex': '$K$ $s^{-1}$'
            },
            'uas': {
                'variable_long_name': 'Eastward Near-Surface Wind',
                'variable_units': 'm s-1',
                'variable_units_latex': '$m$ $s^{-1}$'
            },
            'vas': {
                'variable_long_name': 'Northward Near-Surface Wind',
                'variable_units': 'm s-1',
                'variable_units_latex': '$m$ $s^{-1}$'
            },

            # --- NON CMIP ---
            # 'alb_ground': 'grnd_alb'
            # 'alb_ground_vis': 'grnd_alb_vis',
            'cld': {
                'variable_long_name': 'Cloud Fraction',
                'variable_units': '%',
                'variable_units_latex': '$%$'
            },
            'cldi': {
                'variable_long_name': 'Vertically Integrated Total Cloud Fraction',
                'variable_units': '%',
                'variable_units_latex': '$%$'
            },
            'sst': {
                'variable_long_name': 'Sea Surface Temperature',
                'variable_units': 'K',
                'variable_units_latex': 'K'
            },
            'rlt': {
                'notes': 'Positive downwards',
                'variable_long_name': 'TOA Net Longwave Radiation',
                'variable_units': 'W m-2',
                'variable_units_latex': '$W$ $m^{-2}$'
            },
            'rst': {
                'notes': 'Positive downwards',
                'variable_long_name': 'TOA Net Shortwave Radiation',
                'variable_units': 'W m-2',
                'variable_units_latex': '$W$ $m^{-2}$'
            },
            'rt': {
                'notes': 'Positive downwards',
                'variable_long_name': 'TOA Net Radiation',
                'variable_units': 'W m-2',
                'variable_units_latex': '$W$ $m^{-2}$'
            }
        }

        if v != None and v in all_attrs:
            return all_attrs[v]

        return all_attrs