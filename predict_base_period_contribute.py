#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from joblib import Parallel, delayed
from functools import partial
import joblib
from joblib import Parallel, delayed
import xgboost as xgb
import os
import math
#%%
def get_year_index(index, year):
    return index.year.get_loc(year)

def get_eva_for_model_fixed_window(data_sub, df5_sc, allowed_lens, model_idx):

    x1 = data_sub.iloc[:allowed_lens].values.astype(float)
    y1 = df5_sc.iloc[model_idx, :allowed_lens].values.astype(float)

    if x1.size < 2 or np.allclose(x1, x1[0]) or np.allclose(y1, y1[0]):
        corr_r2 = np.nan
    else:
        corr_raw, _ = pearsonr(x1, y1)
        corr_r2 = float(corr_raw**2)

    mae1 = float(np.mean(np.abs(x1 - y1)))
    rmse1 = float(np.sqrt(mean_squared_error(x1, y1)))

    return {
        'model_idx': int(model_idx),
        'corrs': [corr_r2],
        'maes': [mae1],
        'rmses': [rmse1]
    }


def normalize_results(results, weights=None, method='minmax'):
    """
    results: list of dicts, each dict contains keys:
        - 'model_idx'
        - 'corrs' (scalar)
        - 'maes'  (scalar)
        - 'rmses' (scalar)
    weights: tuple/list for (corr_weight, mae_weight, rmse_weight)
    method: 'minmax' or 'zscore' (currently only minmax used for direction inversion)
    Returns: list of dicts with normalized metrics and combined score
    """
    corrs = np.array([res['corrs'] for res in results], dtype=float)
    maes  = np.array([res['maes']  for res in results], dtype=float)
    rmses = np.array([res['rmses'] for res in results], dtype=float)

    def minmax_norm(x):
        xmin, xmax = np.min(x), np.max(x)
        denom = xmax - xmin
        if denom == 0:
            return np.zeros_like(x, dtype=float)
        return (x - xmin) / denom

    norm_maes = minmax_norm(maes)
    norm_rmses = minmax_norm(rmses)

    w = np.array(weights, dtype=float)
    if np.all(w == 0):
        raise ValueError("At least one weight must be non-zero.")
    w = w / np.sum(np.abs(w))
    
    if weights is None:
        w_cc, w_mae, w_rmse = 1/3, 1/3, 1/3
    else:
        w_cc, w_mae, w_rmse = weights
    
    scores = -corrs * w_cc + norm_maes * w_mae + norm_rmses * w_rmse

    normalized_results = []
    for i in range(len(results)):
        normalized_results.append({
            'model_idx': results[i]['model_idx'],
            'score': float(scores[i])
        })
    return normalized_results

#%%
df5 = pd.read_csv(f'./migrated_500000.csv')

df5.columns = list(range(1979, 2026))
data_all = pd.read_csv('./precursor_factors.csv',index_col=0)
data_all = data_all.drop(columns=['mse_850','vm_eof2','tp_feb'])

range_start = 1985
range_end = 2026

index_sy = pd.date_range(start='1979', end=f'{range_end+1}', freq='YS')

bias_history = []  

all_years_results = []

for end_year in range(range_start, range_end):
    ll = 6
    start_year = end_year - ll
    pf = data_all.shape[1] - 1

    index_y = pd.date_range(start=f'{start_year}', end=f'{end_year+1}', freq='YS')
    start_test1 = get_year_index(index_y, end_year-ll)
    end_test1 = get_year_index(index_y, end_year)
    print(f"Year: {end_year}, Test_sel: {start_test1} to {end_test1}")
    segments = [(start_test1, end_test1)]
    segment_years = [index_y[start:end] for start, end in segments] 
    segment_lens = [end - start for start, end in segments]
    allowed_lens = segment_lens[0]

    # # ----------------- Initial -----------------
    data_sub = data_all['scssm_onset'].loc[start_year:end_year-1]
    df5_s_sub = df5.loc[:, start_year:end_year-1]
    pred_2 = df5_s_sub.loc[:, end_year-ll:end_year-ll]
    true_2 = data_sub.loc[end_year-ll:end_year-ll].values

    threshold = max(1, int(math.ceil(0.1 * df5.shape[0])))

    temp_start_year = start_year
    temp_allowed_lens = allowed_lens

    while temp_allowed_lens > 1 and temp_start_year < end_year:
        print(f"Start: {temp_start_year}, Lens: {temp_allowed_lens}, Threshold: {threshold}")

        data_sg = data_all['scssm_onset'].loc[temp_start_year:end_year-1]
        df5_sg = df5.loc[:, temp_start_year:end_year-1]

        pred_g = df5_sg.loc[:, temp_start_year:temp_start_year]           
        true_g = data_sg.loc[temp_start_year:temp_start_year].values      

        diff_g = pred_g.to_numpy() - true_g
        
        if true_g[0] > 28:
            mask_g = np.isclose(pred_g.to_numpy(), true_g, atol=0.5) & (diff_g >= 0)
            ascending = True
        elif true_g[0] == 28:
            mask_g = np.isclose(pred_g.to_numpy(), true_g, atol=0.5)
            ascending = False
        else:
            mask_g = np.isclose(pred_g.to_numpy(), true_g, atol=0.5) & (diff_g <= 0)
            ascending = False

        b_g = np.flatnonzero(mask_g)   
        c_g = df5.iloc[b_g, :]
        c_sorted = c_g.sort_values(by=temp_start_year, ascending=ascending)


        if c_sorted.shape[0] >= threshold:
            start_year = temp_start_year
            allowed_lens = temp_allowed_lens
            break
        
        temp_start_year += 1
        temp_allowed_lens -= 1


    if temp_allowed_lens == allowed_lens:
        data_sg = data_all['scssm_onset'].loc[start_year:end_year-1]
        df5_sg = df5.loc[:, start_year:end_year-1]
        errors = np.abs(pred_2.values.ravel() - true_2.ravel())
        print("mae:", errors.mean())
        p = 10
        th = np.percentile(errors, p)
        close_models_2 = np.where(errors <= th)[0]
        df5_sc = df5_sg.iloc[close_models_2, :]
    else:
        data_sg = data_all['scssm_onset'].loc[start_year:end_year-1]
        df5_sg = df5.loc[:, start_year:end_year-1]
        df5_sg = df5_sg.iloc[:, :allowed_lens]
        errors = np.abs(pred_g.values.ravel() - true_g.ravel())
        print("mae:", errors.mean())
        p = 10
        th = np.percentile(errors, p)
        close_models_2 = np.where(errors <= th)[0]
        df5_sc = df5_sg.iloc[close_models_2, :]

    results = Parallel(n_jobs=24)(delayed(get_eva_for_model_fixed_window)(
    data_sg, df5_sc, allowed_lens, i) for i in range(len(df5_sc)))

    best_result_corr = max(results, key=lambda x: x['corrs'])
    best_result_mae = min(results, key=lambda x: x['maes'])
    best_result_rmse = min(results, key=lambda x: x['rmses'])

    idx_corr = close_models_2[best_result_corr['model_idx']]
    idx_mae = close_models_2[best_result_mae['model_idx']]
    idx_rmse = close_models_2[best_result_rmse['model_idx']]

    #predict
    pre_corr = df5.iloc[idx_corr, get_year_index(index_sy, end_year)]
    pre_mae = df5.iloc[idx_mae, get_year_index(index_sy, end_year)]
    pre_rmse = df5.iloc[idx_rmse, get_year_index(index_sy, end_year)]
    print(f"{end_year} Predicted SCSSM Onset - CC: {pre_corr:.2f}, MAE: {pre_mae:.2f}, RMSE: {pre_rmse:.2f}")
    print('----------------------------------------------------------')

    all_prev = [b for b in bias_history if b['Year'] < end_year]
    prev_biases = sorted(all_prev, key=lambda x: x['Year'], reverse=True)[:ll]  # 取最近 ll 年的偏差记录

    if len(prev_biases) == 0:
        weights_for_ce = None
    else:
        mean_bias_cc = np.sqrt(np.mean([b['Bias_CC_rmse'] for b in prev_biases]))
        mean_bias_mae = np.sqrt(np.mean([b['Bias_MAE_rmse'] for b in prev_biases]))
        mean_bias_rmse = np.sqrt(np.mean([b['Bias_RMSE_rmse'] for b in prev_biases]))
        
        eps = 1e-8
        p_pow = 1
        inv_cc = (1.0 / (mean_bias_cc + eps)) ** p_pow
        inv_mae = (1.0 / (mean_bias_mae + eps)) ** p_pow
        inv_rmse = (1.0 / (mean_bias_rmse + eps)) ** p_pow
        s = inv_cc + inv_mae + inv_rmse
        weights_for_ce = [inv_cc / s, inv_mae / s, inv_rmse / s]

    if weights_for_ce is not None:
        normalized_results = normalize_results(results, weights=weights_for_ce)
        best_result_ce = min(normalized_results, key=lambda x: x['score'])
        ce_idx_local = best_result_ce['model_idx']  
        idx_ce = close_models_2[ce_idx_local]     
        ce_cor = results[ce_idx_local]['corrs']
        ce_mae = results[ce_idx_local]['maes']
        ce_rmse = results[ce_idx_local]['rmses']
    else:
        best_result_ce = None
        idx_ce = None
        ce_cor = ce_mae = ce_rmse = None

    col_pos = get_year_index(index_sy, end_year)
    true_val = data_all.loc[end_year, data_all.columns[5]]
    bias_cc_mae = np.abs(df5.iloc[idx_corr, col_pos] - true_val) if idx_corr is not None else np.nan
    bias_mae_mae = np.abs(df5.iloc[idx_mae, col_pos] - true_val) if idx_mae is not None else np.nan
    bias_rmse_mae = np.abs(df5.iloc[idx_rmse, col_pos] - true_val) if idx_rmse is not None else np.nan
    
    bias_cc_rmse = np.power(df5.iloc[idx_corr, col_pos] - true_val, 2) if idx_corr is not None else np.nan
    bias_mae_rmse = np.power(df5.iloc[idx_mae, col_pos] - true_val, 2) if idx_mae is not None else np.nan
    bias_rmse_rmse = np.power(df5.iloc[idx_rmse, col_pos] - true_val, 2) if idx_rmse is not None else np.nan

    bias_history.append({
        'Year': end_year,
        'Bias_CC_mae': float(bias_cc_mae),
        'Bias_MAE_mae': float(bias_mae_mae),
        'Bias_RMSE_mae': float(bias_rmse_mae),
        'Bias_CC_rmse': float(bias_cc_rmse),
        'Bias_MAE_rmse': float(bias_mae_rmse),
        'Bias_RMSE_rmse': float(bias_rmse_rmse)
    })
    all_years_results.append({
        'end_year': end_year,
        'best_result_corr': best_result_corr,
        'best_result_mae': best_result_mae,
        'best_result_rmse': best_result_rmse,
        'idx_corr': int(idx_corr),
        'idx_mae': int(idx_mae),
        'idx_rmse': int(idx_rmse),
        'best_result_ce': best_result_ce,
        'idx_ce': (int(idx_ce) if idx_ce is not None else None),
        'ce_metrics': {
            'corrs': ce_cor,
            'maes': ce_mae,
            'rmses': ce_rmse
        } if best_result_ce is not None else None
    })
#%%
data_all_2 = data_all.loc[1979:2025]
data_all_2 = data_all_2[data_all_2.columns[0:pf]]
data_all_ss = data_all_2.apply(lambda x:(x-np.mean(x))/np.std(x)).values
data_all_ss = data_all_ss[:,:]

all_years_results_add_sel = [res for res in all_years_results if res['end_year'] in range(1986, 2026)]
idx_ce_list = []
for n in range(len(all_years_results_add_sel)):
    idx_ce_list.append(all_years_results_add_sel[n]['idx_ce'])

model_paths = []
for idx in idx_ce_list:
        model_paths.append(f'./migrated_models/model_migrated_{idx+1}.pkl')

data_all = data_all.rename(columns={'stt': 'STT'})
data_all = data_all.rename(columns={'enso': 'ENSO'})
data_all = data_all.rename(columns={'sam': 'SAM'})
data_all = data_all.rename(columns={'ao_mar': 'AO'})
data_all = data_all.rename(columns={'lst': 'LST'})
#%%
import shap
os.makedirs('./fig', exist_ok=True)

feature_names = list(data_all.columns[:pf])

shap.initjs()

for n in range(1986, 2026):
    idx_1 = n - 1986  
    year = all_years_results_add_sel[idx_1]["end_year"]

    model = joblib.load(model_paths[idx_1])
    booster = model.get_booster()

    sample_np = data_all_ss[idx_1+7:idx_1+8, 0:pf] 
    dm_sample = xgb.DMatrix(sample_np)

    if hasattr(model, "best_iteration") and model.best_iteration is not None:
        limit = model.best_iteration + 1
    else:
        limit = booster.num_boosted_rounds()

    contribs = booster.predict(dm_sample, pred_contribs=True, iteration_range=(0, limit))
    feature_shap = contribs[0, :-1]
    bias = contribs[0, -1]
    
    y_pred = float(model.predict(sample_np))
    reconstructed = float(bias + feature_shap.sum())

    sample_series = pd.Series(sample_np[0], index=feature_names)

    force_vis = shap.force_plot(
        bias,              
        feature_shap,        
        sample_series,       
        matplotlib=False
    )
    html_path = f'../fig/shap_force_{year}.html'
    shap.save_html(html_path, force_vis)
    print(f"Saved interactive force plot: {html_path}")

    exp = shap.Explanation(
        values=feature_shap,    
        base_values=bias,        
        data=sample_np[0],       
        feature_names=feature_names
    )

    plt.figure(figsize=(8, 4), dpi=300)
    png_path = f'./fig/shap_waterfall_{year}.png'
    plt.title(f'Best XG-CE {year}', loc='left', fontsize=12, bbox={'facecolor':'white', 'edgecolor':'black'})
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved static waterfall plot: {png_path}")

#%%
feature_names = list(data_all.columns[:pf])
years = list(range(1986, 2026))
top_k = 8  

records = []

for i, year in enumerate(years):
    model = joblib.load(model_paths[i])
    booster = model.get_booster()

    sample_np = data_all_ss[i+7:i+8, 0:pf]
    dm_sample = xgb.DMatrix(sample_np)
    
    if hasattr(model, "best_iteration") and model.best_iteration is not None:
        limit = model.best_iteration + 1
    else:
        limit = booster.num_boosted_rounds()

    contribs = booster.predict(dm_sample, pred_contribs=True, iteration_range=(0, limit))
    shap_vals = contribs[0, :-1]  # (pf,)
    bias = contribs[0, -1]

    abs_sum = np.sum(np.abs(shap_vals)) + 1e-12
    shares = np.abs(shap_vals) / abs_sum

    rec = {"year": year}
    for f, name in enumerate(feature_names):
        rec[name] = shares[f]
    records.append(rec)

df = pd.DataFrame(records).set_index("year").sort_index()

mean_share = df.mean(axis=0).sort_values(ascending=False)
top_features = list(mean_share.index[:top_k])

df_top = df[top_features].copy()
if len(feature_names) > top_k:
    df_top["other"] = df.drop(columns=top_features).sum(axis=1)


#%%
feature_names = list(data_all.columns[:pf])
years = list(range(1986, 2026))

signed_records = []
abs_share_records = []
signed_share_records = []

for i, year in enumerate(years):
    model = joblib.load(model_paths[i])
    booster = model.get_booster()
    sample_np = data_all_ss[i+7:i+8, 0:pf]
    dm_sample = xgb.DMatrix(sample_np)

    if hasattr(model, "best_iteration") and model.best_iteration is not None:
        limit = model.best_iteration + 1
    else:
        limit = booster.num_boosted_rounds()

    contribs = booster.predict(dm_sample, pred_contribs=True, iteration_range=(0, limit))
    shap_vals = contribs[0, :-1]  # 带符号
    abs_sum = np.sum(np.abs(shap_vals)) + 1e-12

    rec_signed = {"year": year}
    rec_abs_share = {"year": year}
    rec_signed_share = {"year": year}
    for f, name in enumerate(feature_names):
        rec_signed[name] = shap_vals[f]
        rec_abs_share[name] = np.abs(shap_vals[f]) / abs_sum
        rec_signed_share[name] = shap_vals[f] / abs_sum 
    signed_records.append(rec_signed)
    abs_share_records.append(rec_abs_share)
    signed_share_records.append(rec_signed_share)


df_si = pd.DataFrame(signed_records).set_index("year").sort_index()


pred_all_ce = []
for n in range(len(all_years_results_add_sel)):
    year = all_years_results_add_sel[n]['end_year']
    pred_ce = df5.iloc[all_years_results_add_sel[n]['idx_ce'], get_year_index(index_sy, year)]
    print(year)
    print("Predicted CE :", f"{pred_ce:.2f}")
    true_ce = data_all.loc[year, data_all.columns[5]]
    pred_all_ce.append({
        'Year': int(year),
        'PRE': pred_ce,
        'OBS': true_ce
    })
df_pred_ce = pd.DataFrame(pred_all_ce).set_index('Year').sort_index()
# df_pred_ce.to_csv("predictions_ways_1986_2025.csv")
df_signed = df_si.join(df_pred_ce)

#%%
xg = df_signed['PRE']
scssm = df_signed['OBS']
xg_fit = np.polyval(np.polyfit(xg.index, xg.values, 3), xg.index)
fit_sel = scssm.loc['1986':'2025']
scssm_fit = np.polyval(np.polyfit(fit_sel.index, fit_sel.values, 3), fit_sel.index)
#%%
plt.rcParams.update({
    'axes.linewidth': 2,
    'font.size': 10,
    'xtick.major.width': 2,
    'ytick.major.width': 2,
    'font.serif': 'Calibri',
})


cols = df_signed.columns.tolist()[:7]
n_rows = 6  

fig, axes = plt.subplots(n_rows, 1, figsize=(6.4, 1.2*n_rows), sharex=True, dpi=300)
fig.subplots_adjust(hspace=0.1) 

years = df_signed.index

for i in range(5):
    name = cols[i]
    ax = axes[i]
    vals = df_signed[name]
    colors = ["#fa930d" if v > 0 else "#1278f4" for v in vals.values]

    ax.bar(years, vals.values, color=colors, width=1, alpha=0.8, edgecolor="black", linewidth=0.5, zorder=3)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlim(years.min()-1, years.max()+1)
    ax.set_ylim(-3.5, 3.5)
    
    if i < n_rows - 1:
        ax.tick_params(axis='x', which='both', labelbottom=False)
    
    if i == 0:
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(True)
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    else:
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='x', which='both', bottom=False, top=False)
    

    label = f"{chr(97 + i)}" 
    ax.text(-0.1, 1.02, label, transform=ax.transAxes, fontsize=12, 
            fontweight='bold', va='top', ha='left', color='black',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.5))


ax_last = axes[-1]
name5, name6 = cols[5], cols[6]
vals5 = df_signed[name5]
vals6 = df_signed[name6]


colors5 = ["#fa930d" if v >= 28.25 else "#1278f4" for v in vals5.values]
ax_last.scatter(years, vals5.values, c=colors5, s=38, alpha=0.6, marker='x', linewidths=1)
ax_last.plot(years, xg_fit, color="#fa930d", linewidth=2, linestyle='-', label='XG-CE Fit (3rd Poly)', zorder=1, alpha=0.5)


colors6 = ["#fa930d" if v >= 28.25 else "#1278f4" for v in vals6.values]
ax_last.scatter(years, vals6.values, c=colors6, s=38, alpha=0.6, marker='o', edgecolors="None", linewidths=1)
ax_last.plot(years, scssm_fit, color="#1278f4", linewidth=2, linestyle='-', label='SCSSM Fit (3rd Poly)', zorder=1, alpha=0.3)

ax_last.fill_between(scssm.index, 20, 37, where=(scssm.index >= 1986) & (scssm.index <= 1991), color='#fa930d', alpha=0.2)
ax_last.text(1987.2, 34, ' +PDO', color='#fa930d', fontsize=8, fontweight='bold')

ax_last.fill_between(scssm.index, 20, 37, where=(scssm.index >= 1997) & (scssm.index <= 2002), color='gray', alpha=0.2)
ax_last.text(1997.8, 34, ' PDO(T)', color='gray', fontsize=8, fontweight='bold')

ax_last.fill_between(scssm.index, 20, 37, where=(scssm.index >= 2007) & (scssm.index <= 2012), color='#1278f4', alpha=0.2)
ax_last.text(2008.2, 34, ' -PDO', color='#1278f4', fontsize=8, fontweight='bold')

ax_last.fill_between(scssm.index, 20, 37, where=(scssm.index >= 2020) & (scssm.index <= 2025), color="#4ce28d", alpha=0.2)
ax_last.text(2021.2, 34, ' Now', color="#4ce28d", fontsize=8, fontweight='bold')

ax_last.axhline(28.25, color="black", linewidth=0.8, ls='--')


from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Obs', markerfacecolor='black', markersize=5, alpha=0.8),
    Line2D([0], [0], marker='x', color='w', label='Pre', markerfacecolor='none', markeredgecolor='black', markersize=3, markeredgewidth=1, alpha=0.8),
    Line2D([0], [0], color='#1278f4', lw=2, label='Fit(O)', alpha=0.5),
    Line2D([0], [0], color='#fa930d', lw=2, label='Fit(P)', alpha=0.5),
    Line2D([0], [0], color='black', lw=1, label='Mean', ls='--'),
]
ax_last.legend(handles=legend_elements, loc="upper right", fontsize=4, ncol=5, frameon=False,
               bbox_to_anchor=(1, 1.15), columnspacing=1, handletextpad=0.38)

ax_last.set_ylim(22, 37)
ax_last.spines['top'].set_visible(False)
ax_last.spines['bottom'].set_visible(True)
ax_last.set_xlabel("Years")

ax_last.text(-0.1, 1.02, "f", transform=ax_last.transAxes, fontsize=12,
             fontweight='bold', va='top', ha='left', color='black',
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.5))


plt.savefig("./feature_signed_lines_1col_5rows_last_two_combined.png", dpi=300, bbox_inches="tight")
plt.show()
#%%
df_mean_5yr = []
for start_year in range(1986, 2026, 5):
    end_year = min(start_year + 4, 2025)
    df_segment = df_si.loc[start_year:end_year]
    mean_vals = df_segment.abs().mean()
    mean_vals['year_range'] = f"{start_year}-{end_year}"
    df_mean_5yr.append(mean_vals)

df_mean_5yr = pd.DataFrame(df_mean_5yr).set_index('year_range')

df_mean_5yr = df_mean_5yr[['ENSO','STT','SAM','LST','AO']]

import matplotlib as mpl
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['font.size'] = 14
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2

fig, ax = plt.subplots(figsize=(12, 5), dpi=300)


n_groups = len(df_mean_5yr.index)      
n_bars = len(df_mean_5yr.columns)      
x = np.arange(n_groups)
bar_width = 0.7 / n_bars             


morandi_colors = plt.cm.tab20.colors  
colors = [morandi_colors[i] for i in range(n_bars)]


for i, col in enumerate(df_mean_5yr.columns):
    ax.bar(
        x + i * bar_width,
        df_mean_5yr[col],
        width=bar_width,
        label=col,
        color=colors[i % len(colors)],
        edgecolor="black",
        linewidth=0.8,
        alpha=0.7
    )


ax.set_xticks(x + bar_width * (n_bars - 1) / 2)
ax.set_xticklabels(df_mean_5yr.index, rotation=0)

ax.axhline(0, color="black", linewidth=0.9)
ax.set_ylabel("Average Signed SHAP Values")
ax.set_ylim(0, 2)
ax.set_xlabel("5-Year Segments")

ax.set_yticks(np.arange(0, 2.1, 0.5))


ax.legend(
    loc='upper right',
    borderaxespad=0.,
    fontsize=14,
    frameon=False,
    ncol=5
)

fig.tight_layout()
fig.savefig("./feature_signed_bar_5yr_segments.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
