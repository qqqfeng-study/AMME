#%%
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import scipy.signal
from scipy.fftpack import fft,ifft
from metpy import constants as mpconsts
from metpy.units import units
import metpy.calc
from metpy.calc import moist_static_energy
from eofs.xarray import Eof
import matplotlib as mpl
#%%precursor factor
sst = xr.open_dataset('../../AOSL_toLYN/AOSLv2025-07/reconstruct/DATA/ERA5_native_SCSSM/SST/ERA5_ecmwf_SST_Y1940-2025_M01-12_monthly.nc').sst.sel(valid_time=slice('1978-01-01','2025-12-31'))
sst = sst.sortby(['longitude', 'latitude'])
sst = sst.drop_vars(['expver','number'])
sst_oni = sst.sel(latitude=slice(-5, 5), longitude=slice(-170, -120))
sst_oni = sst_oni.mean(dim=['latitude', 'longitude'])

base_period = slice('1991-01-01', '2020-12-31')
sst_oni_base = sst_oni.sel(valid_time=base_period).groupby('valid_time.month').mean(dim='valid_time')
sst_oni_anom = sst_oni.groupby('valid_time.month') - sst_oni_base

sst_oni_anom = sst_oni_anom.rolling(valid_time=5, center=True).mean()
sst_oni_anom = sst_oni_anom.sel(valid_time=slice('1979-01-01', '2025-06-30'))
sst_oni_anom_3 = sst_oni_anom.sel(valid_time = sst_oni_anom['valid_time.month'].isin([2,3,4]))
sst_oni_m = sst_oni_anom_3.resample(valid_time='Y').mean()
# #detrend
def detrend_dim(da, dim):
    """Detrend data array along a given dimension."""
    x = np.arange(da[dim].size)
    coeffs = np.polyfit(x, da, deg=1)
    trend = np.polyval(coeffs, x)
    detrended = da - xr.DataArray(trend, coords={dim: da[dim]}, dims=[dim])
    return detrended
sst_oni_detrended = sst_oni_m

sst_oni_detrended = (sst_oni_detrended-sst_oni_detrended.mean())/sst_oni_detrended.std()

#%%

def wrap_lon_to_360(da, lon='lon'):
    '''
    Wrap longitude coordinates of DataArray to 0..359.75

    Parameters
    ----------
    da : DataArray
        object with longitude coordinates
    lon : string
        name of the longitude

    Returns
    -------
    wrapped : DataArray
        Another data array wrapped around.
    '''
    # wrap -180..179.75 to 0..359.75
    da = da.assign_coords(**{lon: ((da[lon] + 360) % 360)})
    # sort the data
    da = da.sortby(lon)
    return da

nino3 = sst.sel(latitude=slice(-5, 5), longitude=slice(-150, -90))
nino3 = nino3.mean(dim=['latitude', 'longitude'])
sst_360 = wrap_lon_to_360(sst, lon='longitude')
nino4 = sst_360.sel(latitude=slice(-5, 5), longitude=slice(160, 180+(180-150)))
nino4 = nino4.mean(dim=['latitude', 'longitude'])
#anomaly
nino3_base = nino3.sel(valid_time=base_period).groupby('valid_time.month').mean(dim='valid_time')
nino4_base = nino4.sel(valid_time=base_period).groupby('valid_time.month').mean(dim='valid_time')
nino3_anom = nino3.groupby('valid_time.month') - nino3_base
nino4_anom = nino4.groupby('valid_time.month') - nino4_base
#%%
nino3_anom_sel = nino3_anom.sel(valid_time=slice('1979-01-01', '2025-06-30'))
nino4_anom_sel = nino4_anom.sel(valid_time=slice('1979-01-01', '2025-06-30'))
nino3_anom_3 = nino3_anom_sel.sel(valid_time = nino3_anom_sel['valid_time.month'].isin([2,3,4]))
nino4_anom_3 = nino4_anom_sel.sel(valid_time = nino4_anom_sel['valid_time.month'].isin([2,3,4]))
nino3_anom_m = nino3_anom_3.resample(valid_time='Y').mean()
nino4_anom_m = nino4_anom_3.resample(valid_time='Y').mean()

data_var = pd.read_csv('./../precursor_factors.csv')
data_var = data_var.set_index(pd.date_range(start='1940', end='2026', freq='Y').year)
lst = data_var[39:].loc[:, 'lst']
stt = data_var[39:].loc[:, 'stt']
sam = data_var[39:].loc[:, 'sam']
ao = data_var[39:].loc[:, 'ao_mar']
scssm = data_var[39:].loc[:, 'scssm_onset']
enso = data_var[39:].loc[:, 'enso']
vm = data_var[39:].loc[:, 'vm_eof2']

lst = (lst - lst.mean()) / lst.std()
stt = (stt - stt.mean()) / stt.std()
sam = (sam - sam.mean()) / sam.std()
ao = (ao - ao.mean()) / ao.std()
scssm = (scssm - scssm.mean()) / scssm.std()
enso = (enso - enso.mean()) / enso.std()
vm = (vm - vm.mean()) / vm.std()


ncep_doe = pd.read_excel('../../AOSL_toLYN/AOSLv2025-07/reconstruct/STT_ENSO_SAM_OD(2017CD).xlsx',header=None)
ncep_doe = ncep_doe.iloc[:, 4]
ncep_doe.index = pd.date_range(start='1979', end='2025', freq='Y').year
ncep_doe[2025] = 30
ncep_doe = (ncep_doe - ncep_doe.mean()) / ncep_doe.std()

p_value_oni_enso = np.corrcoef(-enso, sst_oni_detrended)[0,1]
p_value_enso_nino3 = np.corrcoef(-enso, nino3_anom_m)[0,1]
p_value_enso_nino4 = np.corrcoef(-enso, nino4_anom_m)[0,1]
#%%
plt.rcParams['font.sans-serif'] = ['Calibri']
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['font.size'] = 14
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2

# 颜色（柔和但区分度高）
color_oni   = '#3E62AD'   
color_n3    = '#3CB371'   
color_n4    = '#F4A460'   
color_enso  = '#C0392B'   

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

ax.plot(
    sst_oni_detrended['valid_time'], sst_oni_detrended,
    label='ONI',
    color=color_oni,
    lw=2.5,
    alpha=0.7
)

ax.plot(
    sst_oni_detrended['valid_time'], -enso.values,
    label='ENSO',
    color=color_enso,
    lw=2.5,
    ls='--',
    alpha=0.7
)

ax.plot(
    sst_oni_detrended['valid_time'], nino3_anom_m.values,
    label='Niño-3',
    color=color_n3,
    lw=2.5,
    alpha=0.7
)

ax.plot(
    sst_oni_detrended['valid_time'], nino4_anom_m.values,
    label='Niño-4',
    color=color_n4,
    lw=2.5,
    alpha=0.7
)

ax.text(
    0.5, 0.9,
    f'Corr(ONI, ENSO) = {p_value_oni_enso:.2f}\n'
    f'Corr(ENSO, Niño-3) = {p_value_enso_nino3:.2f}\n'
    f'Corr(ENSO, Niño-4) = {p_value_enso_nino4:.2f}',
    transform=ax.transAxes,
    fontsize=12,
    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.5)
)   

ax.set_ylim(-3, 3)


ax.tick_params(axis='both', which='major', length=6, width=2)
ax.set_xlabel('Year')
ax.set_ylabel('SST anomaly (°C)')


ax.legend(
    loc='upper right',
    frameon=False,
    fontsize=12,
    handlelength=2.8
)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()


#%%
from scipy.stats import pearsonr
data = pd.DataFrame({
    'oni': sst_oni_detrended,
    'enso': enso.values,
    'od': scssm.values,
    'od_ncep': ncep_doe.values,
    'lst': lst.values,
    'stt': stt.values,
    'sam': sam.values,
    'ao': ao.values,
    'vm': vm.values
}, index=pd.date_range(start='1979', end='2026', freq='Y'))
data.index = data.index.year
corr, p_value = pearsonr(data['vm'], data['od'])
print(f"Correlation between ONI and ENSO influence: {corr:.4f}, p-value: {p_value:.4e}")  
# %%
from scipy.stats import t
def moving_correlation(x, y, window_size):
    corrs = []
    half_window = window_size // 2
    for i in range(len(x)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(x), i + half_window + 1)
        if end_idx - start_idx < window_size:
            corrs.append(np.nan)
            continue
        corr, _ = pearsonr(x[start_idx:end_idx], y[start_idx:end_idx])
        corrs.append(corr)
    return np.array(corrs)

def r_critical(n, alpha=0.1):
    df = n - 2
    tcrit = t.ppf(1 - alpha/2, df)
    rcrit = tcrit / np.sqrt(tcrit**2 + df)
    return rcrit


Rcrit_5 = r_critical(5)
Rcrit_17 = r_critical(17)
Rcrit_25 = r_critical(25)

print("Rcrit (n=5) =", Rcrit_5)
print("Rcrit (n=25) =", Rcrit_25)
corrs_enso_ina = moving_correlation(data['od'].values, data['enso'].values, 5)
corrs_stt_ina = moving_correlation(data['od'].values, data['stt'].values, 5)
corrs_sam_ina = moving_correlation(data['od'].values, data['sam'].values, 5)
corrs_ao_ina = moving_correlation(data['od'].values, data['ao'].values, 5)
corrs_lst_ina = moving_correlation(data['od'].values, data['lst'].values, 5)


corrs_enso_ind = moving_correlation(data['od'].values, data['enso'].values, 25)
corrs_stt_ind = moving_correlation(data['od'].values, data['stt'].values, 25)
corrs_sam_ind = moving_correlation(data['od'].values, data['sam'].values, 25)
corrs_ao_ind = moving_correlation(data['od'].values, data['ao'].values, 17)
corrs_lst_ind = moving_correlation(data['od'].values, data['lst'].values, 25)
#%%找到各因子与P值对应的年份
years = data.index
p_years = []
for corrs in [corrs_enso_ind, corrs_stt_ind, corrs_sam_ind]:
    p_year = np.where((np.abs(corrs) <= Rcrit_25))[0][0]
    p_years.append(years[p_year])
    
for corrs in [corrs_lst_ind]:
    p_year = np.where((np.abs(corrs) >= Rcrit_25))[0][0]
    p_years.append(years[p_year])
    
p_years


#%%
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, dpi=300)

ax1, ax2 = axes

plt.rcParams['font.sans-serif'] = ['Calibri']
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['font.size'] = 20
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2

morandi_colors = plt.cm.tab20.colors 

lw = 3
alpha = 1

for i, p_year in enumerate(p_years):
    ax1.axvline(p_year, color=morandi_colors[i], linestyle=':', lw=1.6, alpha=0.9)
    ax1.text(p_year+0.2, -0.65, str(p_year), rotation=0, verticalalignment='bottom', color=morandi_colors[i], fontsize=14,fontweight='bold')

ax1.plot(data.index, corrs_enso_ind,  color=morandi_colors[0], lw=lw)
ax1.plot(data.index, corrs_stt_ind,  color=morandi_colors[1], lw=lw)
ax1.plot(data.index, corrs_sam_ind,  color=morandi_colors[2], lw=lw)
ax1.plot(data.index, corrs_lst_ind,  color=morandi_colors[3], lw=lw)
ax1.plot(data.index, corrs_ao_ind,   color=morandi_colors[4], lw=lw)

ax1.axhline(-Rcrit_25, color="#393333", linestyle='--', lw=1.6)
ax1.text(0.996, 0.52, 'p<0.1 (25yr)', transform=ax1.transAxes, color="#393333", fontsize=14, ha='right', va='top')
ax1.axhline(-Rcrit_17, color="#393333", linestyle='-', lw=1.6)
ax1.text(0.996, 0.42, 'p<0.1 (17yr)', transform=ax1.transAxes, color="#393333", fontsize=14, ha='right', va='top')
ax2.text(-0.05, 1.1, 'a', transform=ax1.transAxes,fontsize=28, fontweight='bold', ha='right', va='top')


ax2.plot(data.index, corrs_enso_ina,  color=morandi_colors[0], lw=lw)
ax2.plot(data.index, corrs_stt_ina,  color=morandi_colors[1], lw=lw)
ax2.plot(data.index, corrs_sam_ina,  color=morandi_colors[2], lw=lw)
ax2.plot(data.index, corrs_lst_ina,  color=morandi_colors[3], lw=lw)
ax2.plot(data.index, corrs_ao_ina,   color=morandi_colors[4], lw=lw)

from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color=morandi_colors[0], lw=lw, label='ENSO'),
    Line2D([0], [0], color=morandi_colors[1], lw=lw, label='STT'),
    Line2D([0], [0], color=morandi_colors[2], lw=lw, label='SAM'),
    Line2D([0], [0], color=morandi_colors[3], lw=lw, label='LST'),
    Line2D([0], [0], color=morandi_colors[4], lw=lw, label='AO'),
]

ax2.legend(
    handles=legend_elements,
    loc='center',
    frameon=False,
    fontsize=14,
    bbox_to_anchor=(1.065, 1)
)

ax2.axhline(-Rcrit_5, color="#393333", linestyle='--', lw=1.6)
ax2.text(0.996, 0.22, 'p<0.1 (5yr)', transform=ax2.transAxes, color="#393333", fontsize=14, ha='right', va='top')
ax2.text(-0.05, 1.1, 'b', transform=ax2.transAxes,fontsize=28, fontweight='bold', ha='right', va='top')
ax2.set_xticks(data.index[::5])
ax2.set_xticklabels(data.index[::5], rotation=0)
plt.subplots_adjust(hspace=0.2, top=0.6, bottom=0.06, left=0.12, right=0.9)
fig.savefig('./fig/precursor_factors_corr.png', dpi=600, bbox_inches='tight')
plt.show()

# %%
