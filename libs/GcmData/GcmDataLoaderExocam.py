from datetime import timedelta
import cftime
import xarray 

from .GcmDataLoader import GcmDataLoader


class GcmDataLoaderExocam(GcmDataLoader):
    def __init__(
        self,
        id,
        path='{id}/atm/hist/{id}.cam.h0.*.nc',
        keep_vars=[
            'gw',
            'hyai',
            'hybi',
            'hyam',
            'hybm',
            'OMEGA',
            'OMEGAT',
            'lev_p',
            'ilev_p',
            'P0',
            'PS',
            'Z3'
        ]
    ):   
        super().__init__(
            id,
            path=path,
            keep_vars=keep_vars
        )

    def load(self):
        data = xarray.open_mfdataset(
            self.path,
            combine='nested',
            concat_dim='time',
            autoclose=True,
            use_cftime=True
        )
        
        # Centre longitude on SS point
        print('Centering longitude')
        data.coords['lon'] = (data.coords['lon'] - 180)

        data = data.assign_attrs({
            'gcm': 'ExoCAM',
            'id': self.id
        })

        # Add lev and ilev coords, in hPa
        data = data.assign(
            lev_p=lambda x: (x['hyam'].lev * x['P0'] + x['hybm'].lev * x['PS']) / 100000,
            ilev_p=lambda x: (x['hyai'].ilev * x['P0'] + x['hybi'].ilev * x['PS']) / 100000
        )

        # Standardise variables and units
        print('Standardising vars')
        data = self.standardise_vars(data)

        return data

    def var_map(self):
        return {
            # ---- Standardise variable names + units to CMIP ----
            # NB: might need to adjust the 86400 value to p_orbit in seconds
            # 'areacella': 'axyp',
            'clivi': lambda x: x['TGCLDIWP'] / 1000.0, # cloud ice water path kg m-2
            'clt': lambda x: x['CLDTOT'] * 100.0,
            'hur': 'RELHUM',
            'hurs': lambda x: x['RELHUM'].isel(lev=(len(x.lev) - 1)),
            'hus': 'Q',
            'huss': lambda x: x['Q'].isel(lev=(len(x.lev) - 1)),
            'evspsbl': lambda x: x['QFLX'],
            'lwp': lambda x: x['TGCLDLWP'] / 1000.0, # liquid water path kg m-2
            'pr': lambda x: x['PRECT'] * 1000.0,
            'prra': lambda x: (x['PRECT'] - (x['PRECSL'] + x['PRECSC'])) * 1000.0,
            'prsn': lambda x: (x['PRECSL'] + x['PRECSC']) * 1000.0,
            'ps': 'PS',
            'rls': 'FLNS',
            'rlds': lambda x: x['FDL'].isel(ilev=(len(x.ilev) - 1)),
            # 'rldscs': 'trdn_grnd_clrsky',
            # 'rlus': 'trup_surf',
            'rlut': 'FLUT', # TOA Outgoing Longwave Radiation
            'rlutcs': 'FLUTC', 
            'rss': 'FSNS',
            'rsscs': 'FSNSC',
            'rsds': 'FSDS', 
            'rsdscs': 'FSDSC', 
            # 'rsus': lambda x: x['incsw_grnd'] - x['srnf_grnd'], 
            # 'rsdt': 'incsw_toa', 
            # 'rsut': lambda x: x['incsw_toa'] - x['srnf_toa'], 
            # 'rsutcs': 'swup_toa_clrsky',
            'sfcWind': 'U10',
            'siconc': lambda x: 100.0 * x['ICEFRAC'],
            # 'sitemptop': 'ts_oice',
            # 'sithick': 'ZSI',
            'sisnthick': 'SNOWHICE', # alt: 'snowdp'
            'snd': 'SNOWHLND', # alt: 'snowdp'
            # 'so': 'sss',
            'sst': 'SST',
            'ta': 'T',
            'tas': lambda x: x['TS'],
            'tauuo': lambda x: x['TAUX'] * -1.0,
            'tauvo': lambda x: x['TAUY'] * -1.0,
            'tntrl': lambda x: x['QRL'] / 86400.0,
            'tntrs': lambda x: x['QRS'] / 86400.0,
            'ua': 'U',
            'uas': lambda x: x['U'].isel(lev=(len(x.lev) - 1)),
            'va': 'V',
            'vas': lambda x: x['V'].isel(lev=(len(x.lev) - 1)),

            # --- NON CMIP ---
            # 'alb_ground': 'grnd_alb',
            # 'alb_ground_vis': 'grnd_alb_vis',
            'cld': lambda x: x['CLOUD'] * 100.0, # cloud fraction, %
            'cldi': lambda x: x['CLDTOT'] * 100.0,
            'rt': lambda x: x['FSNT'] - x['FLNT'],
            'rlt': 'FLNT',
            'rst': 'FSNT',
        }