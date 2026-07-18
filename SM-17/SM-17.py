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
from scipy.signal import butter, filtfilt
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
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
    da = da.assign_coords(**{lon: ((da[lon] + 360) % 360)})
    da = da.sortby(lon)
    return da
def butter_bandpass(ts, low=3.0, high=7.0, fs=1.0, order=4):
    nyq = 0.5 * fs
    lowcut = 1.0 / high / nyq
    highcut = 1.0 / low / nyq
    b, a = butter(order, [lowcut, highcut], btype='band')
    return filtfilt(b, a, ts)
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
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False) 
    filtered_data = filtfilt(b, a, data)
    return filtered_data
def detrend_dim(da, dim, deg=1):
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit
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
sst_oni_detrended = sst_oni_m
sst_oni_detrended = (sst_oni_detrended-sst_oni_detrended.mean())/sst_oni_detrended.std()
#%%
sst_pdo = sst.sel(valid_time=slice(f'1979-01-01','2024-12-31'))
sst_pdo = xr.DataArray(sst_pdo, coords={'valid_time': sst_pdo.valid_time, 'latitude': sst_pdo.latitude, 'longitude': sst_pdo.longitude})
sst_pdo = sst_pdo.rename({'valid_time':'time'})
ssta = sst_pdo.groupby('time.month') - sst_pdo.groupby('time.month').mean('time')
ssta = ssta.resample(time='1Y').mean(dim='time')
ssta = ssta - ssta.weighted(np.cos(np.deg2rad(ssta.latitude))).mean(('latitude','longitude'))
ssta = wrap_lon_to_360(ssta, lon='longitude')
[leftlon, rightlon, lowerlat, upperlat] = [120, 260, 20, 60]
sst_y = ssta.sel(longitude=slice(leftlon, rightlon), latitude=slice(lowerlat, upperlat))
coslat = np.cos(np.deg2rad(sst_y.coords['latitude'].values))
wgts = np.sqrt(coslat)[..., np.newaxis]
solver = Eof(sst_y, weights=wgts,center=True)
eof1 = solver.eofsAsCorrelation().squeeze()
pc = solver.pcs(pcscaling=1).squeeze()
var = solver.varianceFraction()
#%%
from scipy.signal import firwin, filtfilt
window = 11  
cutoff = 1/11  
b = firwin(window, cutoff, pass_zero='lowpass')
pc1 = -pc[:,0]
pc1_low = filtfilt(b, 1, pc1)
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
enso = (enso - enso.mean()) / enso.std()
vm = (vm - vm.mean()) / vm.std()
pc1_low_series = pd.Series(pc1_low, index=data_var[39:-1].index)
#%%
ncep_doe = pd.read_excel('../../AOSL_toLYN/AOSLv2025-07/reconstruct/STT_ENSO_SAM_OD(2017CD).xlsx',header=None)
ncep_doe = ncep_doe.iloc[:, 4]
ncep_doe.index = pd.date_range(start='1979', end='2025', freq='Y').year
ncep_doe[2025] = 30
ncep_doe = (ncep_doe - ncep_doe.mean()) / ncep_doe.std()
#%%
X = np.column_stack([stt[:32], enso[:32], sam[:32]])
Y = scssm[:32]
model = LinearRegression()
model.fit(X, Y)
a, b, c = model.coef_
intercept = model.intercept_
print(f"a = {a:.3f}, b = {b:.3f}, c = {c:.3f}, intercept = {intercept:.3f}")
scssm_lg = a * stt + b * enso + c * sam + intercept
from scipy.stats import pearsonr
corr, p_value = pearsonr(scssm, scssm_lg)
print(f"Correlation between Observed and Predicted SCSSM: {corr:.4f}, p-value: {p_value:.4e}")
rf = [27.68,27.32,27.39,31.01,29.1,29.49,26.83,29.17,27.19,27.15,31.34,26.86,26.71,27.62,27.78,27.86,28.22,27.49,26.77,28.18,29.15,29.08,28.76,26.16,29.74,30.02]
rf = pd.Series(rf, index=scssm.index[21:])
xg = pd.read_csv('./../predictions_ways_1986_2025.csv', index_col=0)
xg = xg['PRE']
corrs_obs_pred_sm = moving_correlation(scssm.values, scssm_lg.values, 19)
corrs_obs_pred_rf = moving_correlation(scssm.values[21:], rf.values, 19)
corrs_obs_pred_xg = moving_correlation(scssm.values[len(scssm)-len(xg):], xg.values, 19)
rcrit_19 = r_critical(19)
tp = np.where((np.abs(corrs_obs_pred_sm) <= rcrit_19))[0]
print("Turning points years:", scssm.index[tp].values)
#%%
def term(name, coef):
    sign = '+' if coef >= 0 else '−'
    return f' {sign} {abs(coef):.2f}*{name}'
#%%
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['font.size'] = 16
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
fig, axes = plt.subplots(
    2, 1,
    figsize=(10, 6),
    sharex=True,
    gridspec_kw={'height_ratios': [0.8, 0.8]},
    dpi=300
)
moranidi_colors = plt.cm.tab20.colors     
color1 = "#1d1d1d"
color2 = moranidi_colors[0]
color3 = moranidi_colors[2]
color4 = moranidi_colors[8]
ax = axes[0]
ax.plot(scssm, color=color1, lw=3, label='Obs',alpha=0.8)
sm_label =(f'OD = {intercept:.2f}' + term('STT', a) + term('ENSO', b) + term('SAM', c))
ax.plot(scssm_lg, color=color2, lw=3, label=sm_label,alpha=0.8)
ax.fill_between(scssm.index, 20, 37, where=(scssm.index >= 1986) & (scssm.index <= 1991), color='#fa930d', alpha=0.2)
ax.text(1987.3, 21, ' +PDO', color='#fa930d', fontsize=12,fontweight='bold')
ax.fill_between(scssm.index, 20, 37, where=(scssm.index >= 1997) & (scssm.index <= 2002), color='gray', alpha=0.2)
ax.text(1997.3, 21, ' PDO Trans', color='gray', fontsize=12,fontweight='bold')
ax.fill_between(scssm.index, 20, 37, where=(scssm.index >= 2007) & (scssm.index <= 2012), color='#1278f4', alpha=0.2)
ax.text(2008.3, 21, ' -PDO', color='#1278f4', fontsize=12,fontweight='bold')
ax.legend(loc='upper right', fontsize=10,ncol=4, frameon=False)
ax.grid(alpha=0.3, linestyle='--')
ax.set_ylim(20,37)
ax.set_xticks(scssm.index[::5])
ax.tick_params(axis='x', rotation=0)
ax = axes[1]
ax.plot(scssm.index, corrs_obs_pred_sm, color=color2, lw=3, label='SM-17')
ax.axhline(rcrit_19, color=color1, linestyle='--', lw=2, label='90%')
ax.legend(loc='upper right', fontsize=10,ncol=4, frameon=False)
ax.grid(alpha=0.3, linestyle='--')
ax.set_xlabel('Years')
ax.set_ylim(0.1, 1.1)
axes[0].text(0.01, 0.9, '(a)', transform=axes[0].transAxes,fontsize=16, fontweight='bold')
axes[1].text(0.01, 0.9, '(b)', transform=axes[1].transAxes,fontsize=16, fontweight='bold')
plt.tight_layout()
plt.subplots_adjust(hspace=0.1, top=0.95, bottom=0.1, left=0.1, right=0.95)
plt.show()
#%%
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['font.size'] = 14
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
fig, axes = plt.subplots(
    2, 1,
    figsize=(10, 6),
    sharex=True,
    gridspec_kw={'height_ratios': [0.8, 0.8]},
    dpi=300
)
moranidi_colors = plt.cm.tab20.colors     
color1 = "#1d1d1d"
color2 = moranidi_colors[0]
color3 = moranidi_colors[2]
color4 = moranidi_colors[8]
ax = axes[0]
ax.plot(scssm, color=color1, lw=3, label='Obs',alpha=0.8)
sm_label =(f'OD = {intercept:.2f}' + term('STT', a) + term('ENSO', b) + term('SAM', c))
ax.plot(scssm_lg, color=color2, lw=3, label=sm_label,alpha=0.8)
ax.fill_between(scssm.index, 20, 37, where=(scssm.index >= 1986) & (scssm.index <= 1991), color='#fa930d', alpha=0.2)
ax.text(1986.8, 21, ' +PDO', color='#fa930d', fontsize=12,fontweight='bold')
ax.fill_between(scssm.index, 20, 37, where=(scssm.index >= 1997) & (scssm.index <= 2002), color='gray', alpha=0.2)
ax.text(1996.2, 21, ' PDO Trans', color='gray', fontsize=12,fontweight='bold')
ax.fill_between(scssm.index, 20, 37, where=(scssm.index >= 2007) & (scssm.index <= 2012), color='#1278f4', alpha=0.2)
ax.text(2007.8, 21, ' -PDO', color='#1278f4', fontsize=12,fontweight='bold')
ax.fill_between(scssm.index, 20, 37, where=(scssm.index >= 2020) & (scssm.index <= 2025), color="#4ce28d", alpha=0.2)
ax.text(2021.3, 21, ' Now', color="#4ce28d", fontsize=12,fontweight='bold')
ax.legend(loc='upper right', fontsize=10,ncol=4, frameon=False)
ax.grid(alpha=0.3, linestyle='--')
ax.set_ylim(20,37)
ax.set_xticks(scssm.index[::5])
ax.tick_params(axis='x', rotation=0)
ax = axes[1]
ax.plot(scssm.index, corrs_obs_pred_sm, color=color2, lw=3, label='SM-17')
ax.axhline(rcrit_19, color=color1, linestyle='--', lw=2, label='90%')
ax.legend(loc='upper right', fontsize=10,ncol=4, frameon=False)
ax.grid(alpha=0.3, linestyle='--')
ax.set_xlabel('Years')
ax.set_ylim(0.1, 1.1)
axes[0].text(-0.08, 1.04, 'a', transform=axes[0].transAxes,fontsize=18, fontweight='bold')
axes[1].text(-0.08, 1.02, 'b', transform=axes[1].transAxes,fontsize=18, fontweight='bold')
plt.tight_layout()
plt.subplots_adjust(hspace=0.2, top=0.95, bottom=0.1, left=0.1, right=0.95)
fig.savefig('./fig/SCSSM_Obs_Pred_Corr.png', dpi=600, bbox_inches='tight')
plt.show()
#%%
plt.figure(figsize=(10, 6))
plt.plot(sst_oni_detrended['valid_time'], sst_oni_detrended, label='ONI', color='b', marker='o')
plt.plot(sst_oni_detrended['valid_time'], -enso.values, label='ENSO', color='r', marker='o')
plt.ylim(-3, 3)
plt.legend()
plt.show()
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
corr, p_value = pearsonr(data['enso'], data['od'])
print(f"Correlation between ONI and ENSO influence: {corr:.4f}, p-value: {p_value:.4e}")  
Rcrit_17 = r_critical(17)
Rcrit_25 = r_critical(25)
print("Rcrit (n=21) =", Rcrit_25)
corrs_enso_ina = moving_correlation(data['od'].values, data['enso'].values, 7)
corrs_stt_ina = moving_correlation(data['od'].values, data['stt'].values, 7)
corrs_sam_ina = moving_correlation(data['od'].values, data['sam'].values, 7)
corrs_ao_ina = moving_correlation(data['od'].values, data['ao'].values, 7)
corrs_lst_ina = moving_correlation(data['od'].values, data['lst'].values, 7)
corrs_enso_ind = moving_correlation(data['od'].values, data['enso'].values, 25)
corrs_stt_ind = moving_correlation(data['od'].values, data['stt'].values, 25)
corrs_sam_ind = moving_correlation(data['od'].values, data['sam'].values, 25)
corrs_ao_ind = moving_correlation(data['od'].values, data['ao'].values, 17)
corrs_lst_ind = moving_correlation(data['od'].values, data['lst'].values, 25)
fit_sel_1 = scssm.loc['1979':'1999']
scssm_fit_1 = np.polyval(np.polyfit(fit_sel_1.index, fit_sel_1.values, 1), fit_sel_1.index)
fit_sel_2 = scssm.loc['2000':'2014']
scssm_fit_2 = np.polyval(np.polyfit(fit_sel_2.index, fit_sel_2.values, 1), fit_sel_2.index)
fit_sel_3 = scssm.loc['2015':'2025']
scssm_fit_3 = np.polyval(np.polyfit(fit_sel_3.index, fit_sel_3.values, 1), fit_sel_3.index)
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
positive_phase = pc1_low_series > 0
negative_phase = pc1_low_series < 0
sel = pc1_low_series
#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': [1, 1]}, dpi=300)
ax1, ax2 = axes
plt.rcParams['font.sans-serif'] = ['Calibri']
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['font.size'] = 16
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
morandi_colors = plt.cm.tab20.colors     
lw = 3
alpha = 1
for i, p_year in enumerate(p_years):
    ax1.axvline(p_year, color=morandi_colors[i], linestyle=':', lw=1.6, alpha=0.9)
    ax1.text(p_year+0.2, -0.65, str(p_year), rotation=0, verticalalignment='bottom', color=morandi_colors[i], fontsize=14,fontweight='bold')
ax1.plot(data.index, corrs_enso_ind, label='ENSO',  color=morandi_colors[0], lw=lw)
ax1.plot(data.index, corrs_stt_ind,  label='STT',   color=morandi_colors[1], lw=lw)
ax1.plot(data.index, corrs_sam_ind,  label='SAM',   color=morandi_colors[2], lw=lw)
ax1.plot(data.index, corrs_lst_ind,  label='LST',   color=morandi_colors[3], lw=lw)
ax1.plot(data.index, corrs_ao_ind,   label='AO',    color=morandi_colors[4], lw=lw)
ax1.axhline(-Rcrit_25, color="#393333", linestyle='--', lw=1.6)
ax1.axhline(-Rcrit_17, color="#393333", linestyle='-', lw=1.6)
ax1.text(data.index[0]-0.4, -Rcrit_25+0.02, f'p<0.1(25)', color="#393333", fontsize=14)
ax1.text(data.index[0]-0.4, -Rcrit_17+0.02, f'p<0.1(17)', color="#393333", fontsize=14)
ax1.set_ylabel('Correlation')
ax1.set_title('(a) Interdecadal Correlation ', fontsize=20, fontweight='bold', loc='left',pad=8)
ax1.legend(bbox_to_anchor=(1, 1), loc='best', frameon=False, fontsize=9)
bar_width = pd.Timedelta(days=350)
colors = np.where(positive_phase, "#efde46d9", "#168bc5")  
bar_width = 1
bars = ax2.bar(sel.index, sel.values, width=bar_width, color=colors, edgecolor="#2D2C2D", linewidth=1, align='center', alpha=0.8, zorder=2)
ax2.axhline(0, color='k', linestyle='-', linewidth=0.8, zorder=1, alpha=0.7)
ax2.set_xlim(data.index[0]-1, data.index[-1]+1)
ax2.set_yticks(np.arange(-2, 3, 1))
ax2.set_yticklabels([str(y) for y in np.arange(-2, 3, 1)])
ax2.set_ylim(-2.2, 2.2)
ax2.plot(data.index, scssm.values, color="#373636", lw=lw, label='Onset')
ax2.plot(fit_sel_1.index, scssm_fit_1, label='T 79-99', color="#f43c27", lw=lw, linestyle='--')
ax2.plot(fit_sel_2.index, scssm_fit_2, label='T 00-14', color="#e79f24", lw=lw, linestyle='--')
ax2.plot(fit_sel_3.index, scssm_fit_3, label='T 15-25', color="#2b41e6", lw=lw, linestyle='--')
ax2.set_ylabel('Normalized Index')
ax2.set_title('(b) Trends in Different PDO Periods', fontsize=20, fontweight='bold', loc='left',pad=8)
ax2.legend(loc='upper right', frameon=False, fontsize=9)
ax2.set_xticks(data.index[::5])
ax2.set_xticklabels(data.index[::5], rotation=0)
plt.show()
