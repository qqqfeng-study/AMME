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
from scipy.stats import pearsonr
import matplotlib as mpl
import seaborn as sns
import cmaps
import matplotlib.colors as mcolors
cmap_design = mpl.colormaps.get_cmap(cmaps.BlueWhiteOrangeRed)
warm = mcolors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap_design.name, a=0, b=1), cmap_design(np.linspace(0.53, 0.9, 256))[0:250], N=255)
cold = mcolors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap_design.name, a=0, b=1), cmap_design(np.linspace(0.1, 0.47, 256))[0:250], N=255)
all_colors = np.vstack((cold(np.linspace(0, 1, 256)),warm(np.linspace(0, 1, 256))))
cold_warm = mcolors.LinearSegmentedColormap.from_list('new_cmap', all_colors)
data_var = pd.read_csv('../precursor_factors.csv',index_col=0)
data_var = data_var.drop(columns=['mse_850','vm_eof2','tp_feb'])
data_var = data_var[39:]  
data_var = data_var.rename(columns={'ao_mar':'AO','scssm_onset':'OD','enso':'ENSO','stt':'STT','sam':'SAM','lst':'LST'})
#%%
cols = data_var.columns
r_all = pd.DataFrame(np.zeros((len(cols), len(cols))), columns=cols, index=cols)
p_all = pd.DataFrame(np.zeros((len(cols), len(cols))), columns=cols, index=cols)
r_b = pd.DataFrame(np.zeros((len(cols), len(cols))), columns=cols, index=cols)
p_b = pd.DataFrame(np.zeros((len(cols), len(cols))), columns=cols, index=cols)
r_a = pd.DataFrame(np.zeros((len(cols), len(cols))), columns=cols, index=cols)
p_a = pd.DataFrame(np.zeros((len(cols), len(cols))), columns=cols, index=cols)
r_n = pd.DataFrame(np.zeros((len(cols), len(cols))), columns=cols, index=cols)
p_n = pd.DataFrame(np.zeros((len(cols), len(cols))), columns=cols, index=cols)
#%%
data_c_all = data_var
data_c_b = data_var.loc[1979:2009]
data_c_a = data_var.loc[2010:]
data_c_n = data_var.loc[2016:2025]
for i in cols:
    for j in cols:
        r_0, p_0 = pearsonr(data_c_all[i], data_c_all[j])
        r_1, p_1 = pearsonr(data_c_b[i], data_c_b[j])
        r_2, p_2 = pearsonr(data_c_a[i], data_c_a[j])
        r_3, p_3 = pearsonr(data_c_n[i], data_c_n[j])
        r_all.loc[i, j] = r_0
        r_b.loc[i, j] = r_1
        r_a.loc[i, j] = r_2
        r_n.loc[i, j] = r_3
        p_all.loc[i, j] = p_0
        p_b.loc[i, j] = p_1
        p_a.loc[i, j] = p_2
        p_n.loc[i, j] = p_3
correlation_1 = r_all
correlation_2 = r_b
correlation_3 = r_a
correlation_4 = r_n
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats import pearsonr
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
plt.rcParams['font.sans-serif'] = ['Calibri']
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
def p_to_stars(p):
    if p < 0.001:
        return '****'
    elif p < 0.01:
        return '***'
    elif p < 0.05:
        return '**'
    elif p < 0.1:
        return '*'
    else:
        return ''
def calc_pvals(data, cols):
    n = len(cols)
    pvals = np.ones((n, n))
    for i in range(n):
        for j in range(i, n):
            x = data[cols[i]].values
            y = data[cols[j]].values
            mask = ~np.isnan(x) & ~np.isnan(y)
            if mask.sum() < 2:
                p = 1.0
            else:
                _, p = pearsonr(x[mask], y[mask])
                if np.isnan(p):
                    p = 1.0
            pvals[i, j] = p
            pvals[j, i] = p
    return pvals
def plot_corr(ax, data, corr, title=None, txt_thresh=0.8):
    cols = corr.columns.tolist()
    n = len(cols)
    pvals = calc_pvals(data, cols)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cold_warm,
        vmin=-1, vmax=1,
        center=0,
        square=True,
        linewidths=0.6,
        linecolor="white",
        cbar=False,
        ax=ax
    )
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(cols, rotation=0, ha='center', fontsize=12)
    ax.set_yticklabels(cols, rotation=0, va='center', fontsize=12)
    for i in range(n):
        for j in range(n):
            if j <= i:
                r = corr.iloc[i, j]
                stars = p_to_stars(pvals[i, j])
                ax.text(j + 0.5, i + 0.5, f"{r:.2f}",
                        ha='center', va='center',
                        color='white' if abs(r) >= txt_thresh else 'black',
                        fontsize=12)
                if stars:
                    ax.text(j + 0.5, i + 0.45, stars,
                            ha='center', va='bottom',
                            fontsize=12, fontweight='bold')
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=36, pad=6, fontweight='bold', loc='left',x=-0.2)
fig, axes = plt.subplots(1, 3, figsize=(21, 7), dpi=600)
plot_corr(axes[0], data_c_all, correlation_1, title='a')
plot_corr(axes[1], data_c_b,   correlation_2, title='b')
plot_corr(axes[2], data_c_a,   correlation_3, title='c')
norm = Normalize(vmin=-1, vmax=1)
sm = ScalarMappable(norm=norm, cmap=cold_warm)
sm.set_array([])
cax = fig.add_axes([0.91, 0.2, 0.01, 0.6])  
cbar = fig.colorbar(sm, cax=cax)
ticks = np.arange(-1, 1.01, 0.2)
cbar.set_ticks(ticks)
cbar.set_ticklabels([f"{t:.1f}" for t in ticks])
cbar.ax.tick_params(labelsize=12, width=2, pad=2)
fig.savefig("./fig/correlation_heatmap_3periods.png", dpi=600, bbox_inches="tight")
plt.show()
#%%
plt.rcParams['font.sans-serif'] = ['Calibri']
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['font.size'] = 14
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
data = data_c_all
corr = correlation_1.copy()
cols = corr.columns.tolist()
n = len(cols)
pvals = np.ones((n, n), dtype=float)
for i in range(n):
    for j in range(i, n):
        x = data[cols[i]].values
        y = data[cols[j]].values
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() < 2:
            p = 1.0
            r = 0.0
        else:
            r, p = pearsonr(x[mask], y[mask])
            if np.isnan(r):
                r = 0.0
                p = 1.0
        pvals[i, j] = p
        pvals[j, i] = p
def p_to_stars(p):
    if p < 0.001:
        return '****'
    elif p < 0.01:
        return '***'
    elif p < 0.05:
        return '**'
    elif p < 0.1:
        return '*'
    else:
        return ''
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = cold_warm
fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
sns.heatmap(
    corr,
    mask=mask,
    cmap=cmap,
    vmin=-1, vmax=1,
    center=0,
    annot=False,
    square=True,
    linewidths=0.6,
    linecolor="white",
    cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    ax=ax
)
ax.set_xticks(np.arange(n) + 0.5)
ax.set_yticks(np.arange(n) + 0.5)
ax.set_xticklabels(cols, rotation=0, ha='center', fontsize=12)
ax.set_yticklabels(cols, rotation=0, va='center', fontsize=12)
for i in range(n):
    for j in range(n):
        if j <= i:
            r_val = corr.iloc[i, j]
            p_val = pvals[i, j]
            stars = p_to_stars(p_val)
            ax.text(j + 0.5, i + 0.5, f"{r_val:.2f}",
                    ha='center', va='center',
                    color='white' if abs(r_val) > 0.55 else 'black',
                    fontsize=12)
            if stars:
                ax.text(j + 0.5, i + 0.45, stars,
                        ha='center', va='bottom',
                        color='black', fontsize=12, fontweight='bold')
for spine in ax.spines.values():
    spine.set_visible(False)
    spine.set_linewidth(2)
    spine.set_color("#beb7b7")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
cbar = ax.collections[0].colorbar
cbar.outline.set_linewidth(2)
cbar.ax.yaxis.set_tick_params(width=2, pad=2)
plt.tight_layout()
plt.show()
#%%
plt.rcParams['font.sans-serif'] = ['Calibri']
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['font.size'] = 14
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
data = data_c_b
corr = correlation_2.copy()
cols = corr.columns.tolist()
n = len(cols)
pvals = np.ones((n, n), dtype=float)
for i in range(n):
    for j in range(i, n):
        x = data[cols[i]].values
        y = data[cols[j]].values
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() < 2:
            p = 1.0
            r = 0.0
        else:
            r, p = pearsonr(x[mask], y[mask])
            if np.isnan(r):
                r = 0.0
                p = 1.0
        pvals[i, j] = p
        pvals[j, i] = p
def p_to_stars(p):
    if p < 0.001:
        return '****'
    elif p < 0.01:
        return '***'
    elif p < 0.05:
        return '**'
    elif p < 0.1:
        return '*'
    else:
        return ''
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = cold_warm
fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
sns.heatmap(
    corr,
    mask=mask,
    cmap=cmap,
    vmin=-1, vmax=1,
    center=0,
    annot=False,
    square=True,
    linewidths=0.6,
    linecolor="white",
    cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    ax=ax
)
ax.set_xticks(np.arange(n) + 0.5)
ax.set_yticks(np.arange(n) + 0.5)
ax.set_xticklabels(cols, rotation=0, ha='center', fontsize=12)
ax.set_yticklabels(cols, rotation=0, va='center', fontsize=12)
for i in range(n):
    for j in range(n):
        if j <= i:
            r_val = corr.iloc[i, j]
            p_val = pvals[i, j]
            stars = p_to_stars(p_val)
            ax.text(j + 0.5, i + 0.5, f"{r_val:.2f}",
                    ha='center', va='center',
                    color='white' if abs(r_val) > 0.55 else 'black',
                    fontsize=12)
            if stars:
                ax.text(j + 0.5, i + 0.45, stars,
                        ha='center', va='bottom',
                        color='black', fontsize=12, fontweight='bold')
for spine in ax.spines.values():
    spine.set_visible(False)
    spine.set_linewidth(2)
    spine.set_color("#beb7b7")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
cbar = ax.collections[0].colorbar
cbar.outline.set_linewidth(2)
cbar.ax.yaxis.set_tick_params(width=2, pad=2)
plt.tight_layout()
plt.show()
#%%
plt.rcParams['font.sans-serif'] = ['Calibri']
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['font.size'] = 14
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
data = data_c_a
corr = correlation_3.copy()
cols = corr.columns.tolist()
n = len(cols)
pvals = np.ones((n, n), dtype=float)
for i in range(n):
    for j in range(i, n):
        x = data[cols[i]].values
        y = data[cols[j]].values
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() < 2:
            p = 1.0
            r = 0.0
        else:
            r, p = pearsonr(x[mask], y[mask])
            if np.isnan(r):
                r = 0.0
                p = 1.0
        pvals[i, j] = p
        pvals[j, i] = p
def p_to_stars(p):
    if p < 0.001:
        return '****'
    elif p < 0.01:
        return '***'
    elif p < 0.05:
        return '**'
    elif p < 0.1:
        return '*'
    else:
        return ''
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = cold_warm
fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
sns.heatmap(
    corr,
    mask=mask,
    cmap=cmap,
    vmin=-1, vmax=1,
    center=0,
    annot=False,
    square=True,
    linewidths=0.6,
    linecolor="white",
    cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    ax=ax
)
ax.set_xticks(np.arange(n) + 0.5)
ax.set_yticks(np.arange(n) + 0.5)
ax.set_xticklabels(cols, rotation=0, ha='center', fontsize=12)
ax.set_yticklabels(cols, rotation=0, va='center', fontsize=12)
for i in range(n):
    for j in range(n):
        if j <= i:
            r_val = corr.iloc[i, j]
            p_val = pvals[i, j]
            stars = p_to_stars(p_val)
            ax.text(j + 0.5, i + 0.5, f"{r_val:.2f}",
                    ha='center', va='center',
                    color='white' if abs(r_val) > 0.55 else 'black',
                    fontsize=12)
            if stars:
                ax.text(j + 0.5, i + 0.45, stars,
                        ha='center', va='bottom',
                        color='black', fontsize=12, fontweight='bold')
for spine in ax.spines.values():
    spine.set_visible(False)
    spine.set_linewidth(2)
    spine.set_color("#beb7b7")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
cbar = ax.collections[0].colorbar
cbar.outline.set_linewidth(2)
cbar.ax.yaxis.set_tick_params(width=2, pad=2)
plt.tight_layout()
plt.show()
