from datetime import datetime, timedelta
from pathlib import Path
import cftime
import warnings
import xarray 

from .GcmDataLoader import GcmDataLoader

# Disable warning when using .rename() which removes coord indexes
warnings.filterwarnings('ignore', category=UserWarning)

def set_date(data):
    # Retrieve month + year from filename (format e.g. APR0001) 
    # Although year can be variable in size
    date_mmmy = data.encoding['source'].split('/').pop().split('.')[0]

    # Extract month number from filename
    month_number = 1
    if date_mmmy[0:3] != 'ANN':
        month_number = datetime.strptime(date_mmmy[0:3], '%b').month

    # Convert to time
    time = cftime.DatetimeNoLeap(
        int(date_mmmy[3:]),
        month_number,
        1
    )

    data = data.assign_coords({ 'time': time })

    return data


class GcmDataLoaderRocke3d(GcmDataLoader):
    def __init__(
        self,
        id,
        path='{id}/*.aij{id}.nc',
        path_vert={
            'aijk': '{id}/*.aijk{id}.nc',
            'aijl': '{id}/*.aijl{id}.nc',
        },
        preprocess=set_date,
        standardise_vars_opt=True,
        keep_vars=[
            'pcldt',
            'qatm',
            'q_100',
            'q_300',
            'q_500',
            'q_700',
            'q_850',
            'rh_100',
            'rh_300',
            'rh_500',
            'rh_700',
            'rh_850',
            't_100',
            't_300',
            't_500',
            't_700',
            't_850',
            'topog',
            'z'
        ]
    ):
        self.path_vert = {}
        self.preprocess = preprocess
        self.standardise_vars_opt = standardise_vars_opt
        for k in path_vert:
            self.path_vert[k] = str(path_vert[k]).format(id=id)
        
        super().__init__(
            id,
            path=path,
            keep_vars=keep_vars
        )

    def load(self):
        data = xarray.open_mfdataset(
            self.path,
            autoclose=True,
            combine='nested',
            concat_dim='time',
            preprocess=self.preprocess
            # use_cftime=True
        )

        # Pull out vertical data
        if len(self.path_vert) > 0:
            print('Extracting vertical data')
            try:            
                data_aijk = xarray.open_mfdataset(
                    self.path_vert['aijk'],
                    autoclose=True,
                    combine='nested',
                    concat_dim='time',
                    preprocess=self.preprocess
                    # use_cftime=True
                )

                aijk_vars = ['tb', 'ub', 'vb', 'w', 'z']
                data = data.assign_coords({
                    'lev': data_aijk.plm.values,
                    'level': data_aijk.level.values
                })
                for v in aijk_vars:
                    data_var = data_aijk[v].rename({ 'plm': 'lev' })
                    if v != 'w':
                        data_var = data_var\
                            .interp(
                                **{
                                    'lat2': data_aijk.lat,
                                    'lon2': data_aijk.lon,
                                    'kwargs': { 'fill_value': 'extrapolate' },
                                }
                            )\
                            .drop_vars(('lat', 'lon'))\
                            .rename({
                                'lat2': 'lat',
                                'lon2': 'lon'
                            })
    
                    data[v] = (
                        data_var.dims,
                        data_var.values
                    )

                data_aijl = xarray.open_mfdataset(
                    self.path_vert['aijl'],
                    autoclose=True,
                    combine='nested',
                    concat_dim='time',
                    preprocess=self.preprocess
                )

                aijl_vars = ['q', 'rh']
                for v in aijl_vars:
                    data_var = data_aijl[v].rename({ 'plm': 'lev' }) 
                    data[v] = (
                        data_var.dims,
                        data_var.values
                    )
            except:
                print('Warning: no vertical data found')
                self.path_vert = {}
                

        # Assign time coords
        # print('Assigning time coords')
        # times = [
        #     cftime.datetime(1, 1, 1, calendar='noleap') + timedelta(days=i) for i, d in enumerate(data['time'].values)
        # ]
        # data = data.assign_coords({ 'time': times })
        data = data.sortby('time')
        
        # Centre longitude on SS point
        print('Centering longitude')
        data.coords['lon'] = (data.coords['lon'] % 360) - 180
        data = data.sortby(data.lon)

        data = data.assign_attrs({
            'gcm': 'ROCKE-3D',
            'id': self.id
        })

        # Standardise variables
        if self.standardise_vars_opt:
            print('Standardising vars')
            data = self.standardise_vars(data)

        return data

    def var_map(self):
        vert_var_map = {}
        if len(self.path_vert) > 0:
            vert_var_map = {
                'hur': 'rh', # %
                'hus': 'q',  # kg kg-1
                'ta': 'tb',  # K
                'ua': 'ub',  # ms-1
                'va': 'vb',  # ms-1
                'w': 'w',    # Pa s-1
                'z': 'z'
            }

        return {
            # NB: might need to adjust the 86400 value to p_orbit in seconds
            'areacella': 'axyp',
            'clivi': lambda x: (x['iwp'] / 1000.0) if 'iwp' in x else None, # cloud ice water path kg m-2
            'clt': lambda x: 100.0 * (1.0 - x['clrsky']), # or pcldt
            'hurs': 'RHsurf',
            'huss': lambda x: x['qsurf'] * 0.0001, # 10^-4 g/g => kg/kg
            'evspsbl': lambda x: x['evap'] / 86400.0,
            'evs': lambda x: x['evap_ocn'] / 86400.0,
            'hfls': 'HWV',
            'lwp': lambda x: x['lwp'] / 1000.0, # liquid water path kg m-2
            'pr': lambda x: x['prec'] / 86400.0,
            'prra': lambda x: (x['prec'] - x['snowfall']) / 86400.0,
            'prsn': lambda x: x['snowfall'] / 86400.0, # mm day-1 => mm seconds-1
            'ps': 'prsurf',
            'rls': 'RTSE',
            'rlds': 'trdn_surf',
            'rldscs': 'trdn_grnd_clrsky',
            'rlus': 'trup_surf',
            # 'rlut': lambda x: x[''], # TOA Outgoing Longwave Radiation
            'rlutcs': 'trup_toa_clrsky', 
            'rss': 'srnf_grnd',
            'rsds': 'incsw_grnd', 
            'rsdscs': 'incsw_grnd_clrsky',
            'rsus': lambda x: x['incsw_grnd'] - x['srnf_grnd'], 
            'rsdt': 'incsw_toa', 
            'rsut': lambda x: x['incsw_toa'] - x['srnf_toa'], 
            'rsutcs': 'swup_toa_clrsky',
            'sfcWind': 'wsurf',
            'siconc': lambda x: x['snowicefr'].where(x['ZSI'].notnull()),
            'sitemptop': 'ts_oice',
            'sithick': 'ZSI',
            'sisnthick': lambda x: (x['snowdp'] / 1000.0).where(x['ZSI'].notnull()), # mm H2O -> m H2O
            'snd': 'zsnow', # alt: 'snowdp'
            'so': 'sss',
            'sst': 'sst',
            'tas': lambda x: x['tsurf'] + 273.15,
            'tauuo' : lambda x: x['tauus'] * 0.001,
            'tauvo' : lambda x: x['tauvs'] * 0.001,
            'uas': 'usurf',
            'vas': 'vsurf',

            # --- NON CMIP ---
            'alb_ground': 'grnd_alb',
            'alb_ground_vis': 'grnd_alb_vis',
            'cldi': lambda x: 100.0 - x['clrsky'],
            'rt': lambda x: x['srnf_toa'] + x['trnf_toa'],
            'rlt': lambda x: -1.0 * x['trnf_toa'],
            'rst': 'srnf_toa',
            'ws': 'wsurf',
            # swn_grnd_clrsky, CLR SKY NET SOLAR RADIATION, SRF
            # nt_dse, DRY STATIC ENERGY

            # --- VERTICAL (aijk) ---
            **vert_var_map
        }