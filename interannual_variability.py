#%%
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker  
from scipy.fftpack import fft,ifft
from eofs.xarray import Eof
import sacpy as sp
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import cmaps
from scipy.signal import butter, filtfilt
import matplotlib as mpl
from matplotlib.lines import Line2D
from scipy.stats import f
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
from scipy.signal import butter, filtfilt
def butter_bandpass(ts, low=3.0, high=7.0, fs=1.0, order=4):
    "order:,2-4"
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
#%%
sst = xr.open_dataset('../../AOSL_toLYN/AOSLv2025-07/reconstruct/DATA/ERA5_native_SCSSM/SST/ERA5_ecmwf_SST_Y1940-2025_M01-12_monthly.nc').sst
sst = sst.sortby(['longitude', 'latitude'])
sst = sst.drop_vars(['expver','number'])
sst = sst.rename({'valid_time':'time'})
start_year = 1979
sst = sst.sel(time=slice(f'{start_year}-01-01','2024-12-31'))
#%%
sst = xr.DataArray(sst, coords={'time': sst.time, 'latitude': sst.latitude, 'longitude': sst.longitude})
ssta = sst.groupby('time.month') - sst.groupby('time.month').mean('time')
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
def plot_eof_and_pc(Extend, lon, lat, eof, var, pc, mode=1, years=None, fnFIG=None):
    proj = ccrs.PlateCarree(central_longitude=180)
    fig = plt.figure(figsize=(12, 5), facecolor='w', dpi=300)
    ax1 = fig.add_axes([0.05, 0.1, 0.52, 0.7], projection=proj)
    ax1.set_extent([Extend[0], Extend[1], Extend[2], Extend[3]], crs=ccrs.PlateCarree())
    ax1.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax1.add_feature(cfeature.COASTLINE.with_scale('50m'), alpha=0.7, edgecolor='gray', linewidth=0.7, zorder=2)
    gl = ax1.gridlines(crs=ccrs.PlateCarree(),
                    draw_labels=True,
                    dms=False,
                    x_inline=False,
                    y_inline=False,
                    linewidth=1,
                    linestyle='dotted',
                    color="black",
                    alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False
    gl.rotate_labels = False
    gl.xlocator = cticker.LongitudeLocator(20)
    gl.ylocator = cticker.LatitudeLocator(5)
    ax1.set_title(f'(a) EOF{mode+1}', loc='left', fontsize=15)
    pct = var[mode] * 100 if isinstance(var, (np.ndarray, list)) else var * 100
    ax1.set_title(f'{pct[mode]:.2f}%', loc='right', fontsize=15)
    newcmp = cmaps.BlueYellowRed
    index = [5, 20, 35, 50, 65, 85, 95, 110, 125, 0, 0, 135, 150, 165, 180, 200, 210, 220, 235, 250]
    color_list = [newcmp[i].colors for i in index]
    color_list[9] = [1., 1., 1.]
    color_list[10] = [1., 1., 1.]
    cmap = mcolors.ListedColormap(color_list)
    levels = np.linspace(-1, 1, 21)
    norm = mcolors.BoundaryNorm(levels, cmap.N)
    contourf = ax1.contourf(lon, lat, eof[mode, :, :], levels=levels, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), extend="both")
    cax = fig.add_axes([0.08, 0.01, 0.45, 0.04])  
    cb = plt.colorbar(contourf, cax=cax, orientation="horizontal",
                    ticks=np.linspace(-1, 1, 5),
                    drawedges=True,
                    extendfrac='auto',
                    extendrect=True)
    ax2 = fig.add_axes([0.65, 0.15, 0.35, 0.55])
    ax2.set_title(f'(b) PC{mode+1}', loc='left', fontsize=15)
    if years is None:
        years = np.arange(1950, 1950 + pc.shape[0])
    pc_mode = pc[:, mode] if pc.ndim == 2 else pc
    pc_pos = np.where(pc_mode > 0, pc_mode, np.nan)
    pc_neg = np.where(pc_mode < 0, pc_mode, np.nan)
    ax2.bar(years, pc_pos, color='red', width=1, edgecolor='k')
    ax2.bar(years, pc_neg, color='blue', width=1, edgecolor='k')
    ax2.set_xlim(years[0], years[-1])
    ax2.set_xlabel("Year")
    plt.draw()
    if fnFIG is not None:
        plt.savefig(fnFIG + ".png", bbox_inches='tight')
    plt.show()
Extend = [leftlon, rightlon, lowerlat, upperlat]
lon, lat = np.meshgrid(sst_y.coords['longitude'].values, sst_y.coords['latitude'].values)
years = np.arange(start_year, 2025)
plot_eof_and_pc(Extend, lon, lat, eof1, var, pc, mode=0, years=years, fnFIG="EOF1_SST_PDO")
pc1 = -pc[:,0]
from scipy.signal import firwin, filtfilt
window = 11  
cutoff = 1/11  
b = firwin(window, cutoff, pass_zero='lowpass')
pc1_low = filtfilt(b, 1, pc1)
#%%
index_time = np.arange(start_year, 2025)
pc1_low_series = pd.Series(pc1_low, index=index_time)
positive_phase = pc1_low_series > 0
negative_phase = pc1_low_series < 0
scssm = pd.read_csv('./../precursor_factors.csv',index_col=0)
scssm = scssm.loc[start_year:2025,'scssm_onset']
scssm_2 = pd.Series(scssm.values, index=pd.date_range(start=f'{start_year}', end='2025', freq='YS'))
from scipy import stats
def linear_trend_test(ts):
    y = ts.values
    x = np.arange(len(y))  
    slope, intercept, r, p, stderr = stats.linregress(x, y)
    fit = intercept + slope * x
    return pd.Series(fit, index=ts.index), slope, p
fit_sel_1 = scssm.loc['1979':'2001']
scssm_fit_1, slope1, p1 = linear_trend_test(fit_sel_1)
print("slope1:",slope1,"p1:",p1)
scssm_fit_1 = pd.Series(scssm_fit_1.values, index=pd.date_range(start='1979', end='2001', freq='YS'))
fit_sel_2 = scssm.loc['2001':'2016']
scssm_fit_2, slope2, p2 = linear_trend_test(fit_sel_2)
print("slope2:",slope2,"p2:",p2)
scssm_fit_2 = pd.Series(scssm_fit_2.values, index=pd.date_range(start='2001', end='2016', freq='YS'))
fit_sel_3 = scssm.loc['2016':'2025']
scssm_fit_3, slope3, p3 = linear_trend_test(fit_sel_3)
print("slope3:",slope3,"p3:",p3)
scssm_fit_3 = pd.Series(scssm_fit_3.values, index=pd.date_range(start='2016', end='2025', freq='YS'))
scssm_norm = (scssm - scssm.mean()) / scssm.std()
scssm_interannual = butter_bandpass(scssm_norm.values, low=3.0, high=8.0, fs=1.0, order=4)
sl_year = 11
var7 = pd.Series(scssm_norm.values, index=scssm_norm.index).rolling(window=sl_year, center=True).std()
rate7 = (
    scssm_norm.diff()
    .rolling(window=sl_year, center=True)
    .apply(lambda x: np.sqrt(np.nanmean(x**2)))
)
diff_x = np.abs(np.diff(scssm_norm.values))
win = sl_year  
Fvals = []
pvals = []
years_F = []
for i in range(win, len(diff_x) - win):
    var1 = np.var(diff_x[i-win:i], ddof=1)   
    var2 = np.var(diff_x[i:i+win], ddof=1)   
    F = var2 / var1
    p = 1 - f.cdf(F, win-1, win-1)      
    Fvals.append(F)
    pvals.append(p)
    years_F.append(scssm_norm.index[i])
Fvals = np.array(Fvals)
pvals = np.array(pvals)
years_F = np.array(years_F)
Fcrit_95 = f.ppf(0.9, win-1, win-1)
#%%
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['font.size'] = 16
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
morandi_colors = plt.cm.tab20.colors     
c1 = morandi_colors[0]
c2 = morandi_colors[14]
c3 = morandi_colors[2]
color1 = "#2B2C30"
color2 = color1
color3 = color1
scssm_fit_1 = pd.Series(scssm_fit_1.values, index=fit_sel_1.index)
scssm_fit_2 = pd.Series(scssm_fit_2.values, index=fit_sel_2.index)
scssm_fit_3 = pd.Series(scssm_fit_3.values, index=fit_sel_3.index)
fit_1_mean = scssm_fit_1.mean()
fit_2_mean = scssm_fit_2.mean()
fit_3_mean = scssm_fit_3.mean()
#%%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import seaborn as sns
import colormaps as cs
from scipy.stats import f, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
def calculate_mae_rmse(true_values, predicted_values):
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    return mae, rmse
rf = [27.68,27.32,27.39,31.01,29.1,29.49,26.83,29.17,27.19,27.15,31.34,26.86,26.71,27.62,27.78,27.86,28.22,27.49,26.77,28.18,29.15,29.08,28.76,26.16,29.74,30.02]
rf = rf[-10:]  
sm = pd.read_csv('./../lg_scssm.csv')
sm = sm.set_index(pd.date_range(start='1979', end='2026', freq='Y').year)
sm = sm.squeeze()
xg = pd.read_csv('./../predictions_ways_1986_2025.csv', index_col=0).loc['1986':'2025','PRE']
true = scssm_2.loc['1979':'2025'].values
sm_series = sm.loc[1979:2025]
xg_series = xg
rf_series = pd.Series(rf, index=range(2016, 2026))
true_series = pd.Series(true, index=range(1979, 2026))
all_years = np.arange(1979, 2026)
true_full = true_series.reindex(all_years)
sm_full = sm_series.reindex(all_years)
xg_full = xg_series.reindex(all_years)
rf_full = rf_series.reindex(all_years)
error_sm = np.abs(sm_full - true_full)
error_rf = np.abs(rf_full - true_full)
error_xg = np.abs(xg_full - true_full)
error_matrix = np.vstack([
    error_xg.values,
    error_rf.values,
    error_sm.values,
])
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True,
                          gridspec_kw={'height_ratios': [1.2, 1, 0.8], 'hspace': 0.1})
xlim_left = 1978
xlim_right = 2026
ax_a = axes[0]
ax_a2 = ax_a.twinx()
bar_width = 1
colors = np.where(pc1_low_series.values > 0, "#fa930d", "#1278f4")
ax_a.bar(pc1_low_series.index, pc1_low_series.values, width=bar_width, color=colors, edgecolor="#2D2C2D", linewidth=1, align='center', alpha=0.8, zorder=2)
c2 = plt.cm.tab20.colors[14]
ax_a.plot(scssm_norm.index, scssm_interannual, color=c2, lw=3, label='Bp', alpha=0.8)
ax_a.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.7)
ax_a.set_ylim(-2.3, 2.3)
ax_a2.plot(scssm.index, scssm, color="#111011", lw=3.5, label='Obs', alpha=0.8)
def sig_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.1:
        return "*"  
    else:
        return ""
def fmt(val):
    s = f"{val:.2f}"
    return "0" if s == "0.00" else s
ax_a2.plot(scssm_fit_1.index, scssm_fit_1, '-', lw=3, color="#2B2C30", alpha=0.8)
ax_a2.text(scssm_fit_1.index[-10] - 8.5, scssm_fit_1.iloc[-10] - 7.95,
           f"Mean={fmt(fit_1_mean)}; Slope={fmt(slope1)}{sig_stars(p1)}", 
           color="#2B2C30", fontsize=12, va="center", alpha=0.8)
ax_a2.plot(scssm_fit_2.index, scssm_fit_2, '-', lw=3, color="#2B2C30", alpha=0.8)
ax_a2.text(scssm_fit_2.index[-10] - 4.5, scssm_fit_2.iloc[-10] - 6.65,
           f"Mean={fmt(fit_2_mean)}; Slope={fmt(slope2)}{sig_stars(p2)}", 
           color="#2B2C30", fontsize=12, va="center", alpha=0.8)
ax_a2.plot(scssm_fit_3.index, scssm_fit_3, '-', lw=3, color="#2B2C30", alpha=0.8)
ax_a2.text(scssm_fit_3.index[-10], scssm_fit_3.iloc[-10] - 8.155,
           f"Mean={fmt(fit_3_mean)}; Slope={fmt(slope3)}{sig_stars(p3)}", 
           color="#2B2C30", fontsize=12, va="center", alpha=0.8)
ax_a2.set_yticks(np.arange(20, 37, 5))
ax_a2.set_ylim(20, 36)
ax_a2.hlines(fit_1_mean, scssm_fit_1.index[0], scssm_fit_1.index[-1], colors="#2B2C30", linestyles='--', lw=3, alpha=0.8,zorder=10)
ax_a2.hlines(fit_2_mean, scssm_fit_2.index[0], scssm_fit_2.index[-1], colors="#2B2C30", linestyles='--', lw=3, alpha=0.8,zorder=10)
ax_a2.hlines(fit_3_mean, scssm_fit_3.index[0], scssm_fit_3.index[-1], colors="#2B2C30", linestyles='--', lw=3, alpha=0.8,zorder=10)
current_phase = pc1_low_series.iloc[0] > 0
start_idx = pc1_low_series.index[0]
for i in range(1, len(pc1_low_series)):
    if (pc1_low_series.iloc[i] > 0) != current_phase:
        end_idx = pc1_low_series.index[i]
        color = "#fa930d" if current_phase else '#1278f4'
        ax_a.axvspan(start_idx, end_idx, color=color, alpha=0.05, zorder=0)
        current_phase = pc1_low_series.iloc[i] > 0
        start_idx = end_idx
ax_a.axvspan(start_idx, pc1_low_series.index[-1] + 1, 
             color=('#fa930d' if current_phase else '#1278f4'), alpha=0.05, zorder=0)
legend_elems = [
    Line2D([0], [0], color="#fa930d", lw=6, label='+PDO'),
    Line2D([0], [0], color='#1278f4', lw=6, label='-PDO'),
    Line2D([0], [0], color='#2B2C30', lw=4, linestyle='-', label='Obs'),
    Line2D([0], [0], color="#2B2C30", lw=2, linestyle='-', label='Trend'),
    Line2D([0], [0], color=c2, lw=2, linestyle='-', label='Bandpass'),
    Line2D([0], [0], color="#2B2C30", lw=2, linestyle='--', label='Mean'),
]
ax_a.legend(handles=legend_elems, loc='upper right', bbox_to_anchor=(0.99, 1.02), 
            fontsize=10, frameon=False, ncol=6)
ax_a.set_xlim(xlim_left, xlim_right)
ax_a.text(0.01, 0.9, '(a)', transform=ax_a.transAxes, fontsize=16, fontweight='bold')
ax_b = axes[1]
ax_b2 = ax_b.twinx()
years_2 = scssm_norm.index
ax_b.bar(years_2[1:], diff_x, color="#1365A9", edgecolor="#2D2C2D", width=1, label='|ΔObs|', alpha=0.7)
ax_b.plot(scssm_norm.index, rate7, color=plt.cm.tab20.colors[2], lw=3, label='Rate')
ax_b.set_ylim(0, np.max(diff_x) * 1.2)
ax_b.set_yticks(np.arange(0, np.max(diff_x) * 1.2, 1))
ax_b2.plot(years_F, Fvals, color="#dc4c36", lw=3, label='F-value', alpha=0.8)
ax_b2.axhline(Fcrit_95, color="#dc4c36", ls='--', lw=2, alpha=0.5)
ax_b2.set_ylim(0, np.max(Fvals) * 1.2)
ax_b.set_xlim(xlim_left, xlim_right)
ax_b.text(0.01, 0.9, '(b)', transform=ax_b.transAxes, fontsize=16, fontweight='bold')
hline = ax_b2.axhline(Fcrit_95, color="#dc4c36", ls='--', lw=2, alpha=0.5)
handles = [ax_b.patches[0], ax_b2.lines[0], ax_b.lines[0], hline]
labels = ['|ΔObs|', 'F-value', 'Rate', '90%']
ax_b.legend(handles, labels, loc='upper right', fontsize=10, frameon=False, ncol=4)
ax_c = axes[2]
extent = [all_years[0] - 0.5, all_years[-1] + 0.5, 0, len(error_matrix)]
im = ax_c.imshow(error_matrix, aspect='auto', cmap=cs.orrd_8, extent=extent, origin='lower')
ax_c.set_yticks(np.arange(len(error_matrix)) + 0.5)
ax_c.set_yticklabels(['XG-CE', 'Hu-25', 'SM-17'])
ax_c.set_ylim(0, len(error_matrix))
ax_c.set_xlim(xlim_left, xlim_right)
ax_c.set_xlabel('Year', fontsize=14)
ax_c.set_xticks(np.arange(xlim_left +1, xlim_right + 1, 5))
ax_c.text(0.01, 0.85, '(c)', transform=ax_c.transAxes, fontsize=16, fontweight='bold')
cax = fig.add_axes([0.138, 0.135, 0.09, 0.015])  
cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
ticks = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
labels = ['' if t not in [2, 4, 6] else str(int(t)) for t in ticks]
cbar.set_ticks(ticks)
cbar.set_ticklabels(labels, fontsize=12)
plt.savefig("fig2_v1.png", dpi=300, bbox_inches='tight')
plt.show()
#%%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import seaborn as sns
import colormaps as cs
from scipy.stats import f, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
def calculate_mae_rmse(true_values, predicted_values):
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    return mae, rmse
rf = [27.68,27.32,27.39,31.01,29.1,29.49,26.83,29.17,27.19,27.15,31.34,26.86,26.71,27.62,27.78,27.86,28.22,27.49,26.77,28.18,29.15,29.08,28.76,26.16,29.74,30.02]
rf = rf[-10:]  
sm = pd.read_csv('./../lg_scssm.csv')
sm = sm.set_index(pd.date_range(start='1979', end='2026', freq='Y').year)
sm = sm.squeeze()
xg = pd.read_csv('./../predictions_ways_1986_2025.csv', index_col=0).loc['1986':'2025','PRE']
true = scssm_2.loc['1979':'2025'].values
sm_series = sm.loc[1979:2025]
xg_series = xg
rf_series = pd.Series(rf, index=range(2016, 2026))
true_series = pd.Series(true, index=range(1979, 2026))
all_years = np.arange(1979, 2026)
true_full = true_series.reindex(all_years)
sm_full = sm_series.reindex(all_years)
xg_full = xg_series.reindex(all_years)
rf_full = rf_series.reindex(all_years)
error_sm = np.abs(sm_full - true_full)
error_rf = np.abs(rf_full - true_full)
error_xg = np.abs(xg_full - true_full)
error_matrix = np.vstack([
    error_xg.values,
    error_rf.values,
    error_sm.values,
])
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True,
                          gridspec_kw={'height_ratios': [1.2, 1, 0.8], 'hspace': 0.1})
xlim_left = 1978
xlim_right = 2026
ax_a = axes[0]
ax_a2 = ax_a.twinx()
bar_width = 1
colors = np.where(pc1_low_series.values > 0, "#fa930d", "#1278f4")
ax_a.bar(pc1_low_series.index, pc1_low_series.values, width=bar_width, color=colors, edgecolor="#2D2C2D", linewidth=1, align='center', alpha=0.8, zorder=2)
c2 = plt.cm.tab20.colors[14]
ax_a.plot(scssm_norm.index, scssm_interannual, color=c2, lw=3, label='Bp', alpha=0.8)
ax_a.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.7)
ax_a.set_ylim(-2.3, 2.3)
ax_a2.plot(scssm.index, scssm, color="#111011", lw=3.5, label='Obs', alpha=0.8)
def sig_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.1:
        return "*"  
    else:
        return ""
def fmt(val):
    s = f"{val:.2f}"
    return "0" if s == "0.00" else s
ax_a2.plot(scssm_fit_1.index, scssm_fit_1, '-', lw=3, color="#2B2C30", alpha=0.8)
ax_a2.text(scssm_fit_1.index[-10] - 8.5, scssm_fit_1.iloc[-10] - 7.95,
           f"Mean={fmt(fit_1_mean)}; Slope={fmt(slope1)}{sig_stars(p1)}", 
           color="#2B2C30", fontsize=12, va="center", alpha=0.8)
ax_a2.plot(scssm_fit_2.index, scssm_fit_2, '-', lw=3, color="#2B2C30", alpha=0.8)
ax_a2.text(scssm_fit_2.index[-10] - 4.5, scssm_fit_2.iloc[-10] - 6.65,
           f"Mean={fmt(fit_2_mean)}; Slope={fmt(slope2)}{sig_stars(p2)}", 
           color="#2B2C30", fontsize=12, va="center", alpha=0.8)
ax_a2.plot(scssm_fit_3.index, scssm_fit_3, '-', lw=3, color="#2B2C30", alpha=0.8)
ax_a2.text(scssm_fit_3.index[-10], scssm_fit_3.iloc[-10] - 8.155,
           f"Mean={fmt(fit_3_mean)}; Slope={fmt(slope3)}{sig_stars(p3)}", 
           color="#2B2C30", fontsize=12, va="center", alpha=0.8)
ax_a2.set_yticks(np.arange(20, 37, 5))
ax_a2.set_ylim(20, 36)
ax_a2.hlines(fit_1_mean, scssm_fit_1.index[0], scssm_fit_1.index[-1], colors="#2B2C30", linestyles='--', lw=3, alpha=0.8,zorder=10)
ax_a2.hlines(fit_2_mean, scssm_fit_2.index[0], scssm_fit_2.index[-1], colors="#2B2C30", linestyles='--', lw=3, alpha=0.8,zorder=10)
ax_a2.hlines(fit_3_mean, scssm_fit_3.index[0], scssm_fit_3.index[-1], colors="#2B2C30", linestyles='--', lw=3, alpha=0.8,zorder=10)
current_phase = pc1_low_series.iloc[0] > 0
start_idx = pc1_low_series.index[0]
for i in range(1, len(pc1_low_series)):
    if (pc1_low_series.iloc[i] > 0) != current_phase:
        end_idx = pc1_low_series.index[i]
        color = "#fa930d" if current_phase else '#1278f4'
        ax_a.axvspan(start_idx, end_idx, color=color, alpha=0.05, zorder=0)
        current_phase = pc1_low_series.iloc[i] > 0
        start_idx = end_idx
ax_a.axvspan(start_idx, pc1_low_series.index[-1] + 1, 
             color=('#fa930d' if current_phase else '#1278f4'), alpha=0.05, zorder=0)
legend_elems = [
    Line2D([0], [0], color="#fa930d", lw=6, label='+PDO'),
    Line2D([0], [0], color='#1278f4', lw=6, label='-PDO'),
    Line2D([0], [0], color='#2B2C30', lw=4, linestyle='-', label='Obs'),
    Line2D([0], [0], color="#2B2C30", lw=2, linestyle='-', label='Trend'),
    Line2D([0], [0], color=c2, lw=2, linestyle='-', label='Bandpass'),
    Line2D([0], [0], color="#2B2C30", lw=2, linestyle='--', label='Mean'),
]
ax_a.legend(handles=legend_elems, loc='upper right', bbox_to_anchor=(0.99, 1.02), 
            fontsize=10, frameon=False, ncol=6)
ax_a.set_xlim(xlim_left, xlim_right)
ax_a.text(-0.06, 1.05, 'a', transform=ax_a.transAxes, fontsize=22, 
          fontweight='bold', va='top', ha='right')
ax_b = axes[1]
ax_b2 = ax_b.twinx()
years_2 = scssm_norm.index
ax_b.bar(years_2[1:], diff_x, color="#1365A9", edgecolor="#2D2C2D", width=1, label='|ΔObs|', alpha=0.7)
ax_b.plot(scssm_norm.index, rate7, color=plt.cm.tab20.colors[2], lw=3, label='Rate')
ax_b.set_ylim(0, np.max(diff_x) * 1.2)
ax_b.set_yticks(np.arange(0, np.max(diff_x) * 1.2, 1))
ax_b2.plot(years_F, Fvals, color="#dc4c36", lw=3, label='F-value', alpha=0.8)
ax_b2.axhline(Fcrit_95, color="#dc4c36", ls='--', lw=2, alpha=0.5)
ax_b2.set_ylim(0, np.max(Fvals) * 1.2)
ax_b.set_xlim(xlim_left, xlim_right)
ax_b.text(-0.06, 1.05, 'b', transform=ax_b.transAxes, fontsize=22, 
          fontweight='bold', va='top', ha='right')
hline = ax_b2.axhline(Fcrit_95, color="#dc4c36", ls='--', lw=2, alpha=0.5)
handles = [ax_b.patches[0], ax_b2.lines[0], ax_b.lines[0], hline]
labels = ['|ΔObs|', 'F-value', 'Rate', '90%']
ax_b.legend(handles, labels, loc='upper right', fontsize=10, frameon=False, ncol=4)
ax_c = axes[2]
extent = [all_years[0] - 0.5, all_years[-1] + 0.5, 0, len(error_matrix)]
im = ax_c.imshow(error_matrix, aspect='auto', cmap=cs.orrd_8, extent=extent, origin='lower')
ax_c.set_yticks(np.arange(len(error_matrix)) + 0.5)
ax_c.set_yticklabels(['XG-CE', 'Hu-25', 'SM-17'], fontsize=14)
ax_c.set_ylim(0, len(error_matrix))
ax_c.set_xlim(xlim_left, xlim_right)
ax_c.set_xlabel('Year', fontsize=14)
ax_c.set_xticks(np.arange(xlim_left +1, xlim_right + 1, 5))
ax_c.text(-0.06, 1.05, 'c', transform=ax_c.transAxes, fontsize=22, 
          fontweight='bold', va='top', ha='right')
cax = fig.add_axes([0.138, 0.135, 0.09, 0.015])  
cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
ticks = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
labels = ['' if t not in [2, 4, 6] else str(int(t)) for t in ticks]
cbar.set_ticks(ticks)
cbar.set_ticklabels(labels, fontsize=12)
plt.savefig("fig2_v1.png", dpi=300, bbox_inches='tight')
plt.show()
