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
import matplotlib as mpl
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
df5 = pd.read_csv(f'../final/migrated_500000.csv')
df5.columns = list(range(1979, 2026))
data_all = pd.read_csv('../precursor_factors.csv',index_col=0)
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
    print(f": {end_year}, : {start_test1}  {end_test1}")
    segments = [(start_test1, end_test1)]
    segment_years = [index_y[start:end] for start, end in segments] 
    segment_lens = [end - start for start, end in segments]
    allowed_lens = segment_lens[0]
    data_sub = data_all['scssm_onset'].loc[start_year:end_year-1]
    df5_s_sub = df5.loc[:, start_year:end_year-1]
    pred_2 = df5_s_sub.loc[:, end_year-ll:end_year-ll]
    true_2 = data_sub.loc[end_year-ll:end_year-ll].values
    threshold = max(1, int(math.ceil(0.1 * df5.shape[0])))
    temp_start_year = start_year
    temp_allowed_lens = allowed_lens
    while temp_allowed_lens > 1 and temp_start_year < end_year:
        print(f": {temp_start_year}, : {temp_allowed_lens}, : {threshold}")
        data_sg = data_all['scssm_onset'].loc[temp_start_year:end_year-1]
        df5_sg = df5.loc[:, temp_start_year:end_year-1]
        pred_g = df5_sg.loc[:, temp_start_year:temp_start_year]            
        true_g = data_sg.loc[temp_start_year:temp_start_year].values       
        diff_g = pred_g.to_numpy() - true_g
        if true_g[0] > 28:
            mask_g = np.isclose(pred_g.to_numpy(), true_g, atol=0.5) & (diff_g >= 0)
            ascending = True
            print("28")
        elif true_g[0] == 28:
            mask_g = np.isclose(pred_g.to_numpy(), true_g, atol=0.5)
            ascending = False
            print("28")
        else:
            mask_g = np.isclose(pred_g.to_numpy(), true_g, atol=0.5) & (diff_g <= 0)
            ascending = False
            print("28")
        b_g = np.flatnonzero(mask_g)     
        c_g = df5.iloc[b_g, :]
        c_sorted = c_g.sort_values(by=temp_start_year, ascending=ascending)
        print(":", c_sorted.shape[0], ":", threshold)
        if c_sorted.shape[0] >= threshold:
            start_year = temp_start_year
            allowed_lens = temp_allowed_lens
            print(f": start_year={start_year}, allowed_lens={allowed_lens}")
            break
        temp_start_year += 1
        temp_allowed_lens -= 1
    if temp_allowed_lens <= 1 or temp_start_year >= end_year:
        print(" atol")
    if temp_allowed_lens == allowed_lens:
        data_sg = data_all['scssm_onset'].loc[start_year:end_year-1]
        df5_sg = df5.loc[:, start_year:end_year-1]
        errors = np.abs(pred_2.values.ravel() - true_2.ravel())
        print("mae:", errors.mean())
        p = 10
        th = np.percentile(errors, p)
        close_models_2 = np.where(errors <= th)[0]
        if close_models_2.size == 0:
            print(f"{end_year}: {p} (={th:.4f})")
        else:
            print(f"{end_year}: {p}={th:.4f} {len(close_models_2)} ")
        df5_sc = df5_sg.iloc[close_models_2, :]
    else:
        data_sg = data_all['scssm_onset'].loc[start_year:end_year-1]
        df5_sg = df5.loc[:, start_year:end_year-1]
        df5_sg = df5_sg.iloc[:, :allowed_lens]
        errors = np.abs(pred_g.values.ravel() - true_g.ravel())
        print(errors[0])
        print("mae:", errors.mean())
        p = 10
        th = np.percentile(errors, p)
        close_models_2 = np.where(errors <= th)[0]
        if close_models_2.size == 0:
            print(f"{end_year}: {p} (={th:.4f})")
        else:
            print(f"{end_year}: {p}={th:.4f} {len(close_models_2)} ")
        df5_sc = df5_sg.iloc[close_models_2, :]
    results = Parallel(n_jobs=24)(delayed(get_eva_for_model_fixed_window)(
    data_sg, df5_sc, allowed_lens, i) for i in range(len(df5_sc)))
    best_result_corr = max(results, key=lambda x: x['corrs'])
    best_result_mae = min(results, key=lambda x: x['maes'])
    best_result_rmse = min(results, key=lambda x: x['rmses'])
    idx_corr = close_models_2[best_result_corr['model_idx']]
    idx_mae = close_models_2[best_result_mae['model_idx']]
    idx_rmse = close_models_2[best_result_rmse['model_idx']]
    pre_corr = df5.iloc[idx_corr, get_year_index(index_sy, end_year)]
    pre_mae = df5.iloc[idx_mae, get_year_index(index_sy, end_year)]
    pre_rmse = df5.iloc[idx_rmse, get_year_index(index_sy, end_year)]
    print(f"{end_year} Predicted SCSSM Onset - CC: {pre_corr:.2f}, MAE: {pre_mae:.2f}, RMSE: {pre_rmse:.2f}")
    print('----------------------------------------------------------')
    all_prev = [b for b in bias_history if b['Year'] < end_year]
    prev_biases = sorted(all_prev, key=lambda x: x['Year'], reverse=True)[:ll]
    if len(prev_biases) == 0:
        weights_for_ce = None
        print(f"{end_year}:  CE  CE")
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
        print(f"{end_year}:  CE  = {weights_for_ce} (mean_biases: {mean_bias_cc:.6f}, {mean_bias_mae:.6f}, {mean_bias_rmse:.6f})")
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
    print(f"{end_year}:  (CC, MAE, RMSE) = ({bias_cc_mae:.6f}, {bias_mae_mae:.6f}, {bias_rmse_mae:.6f})")
    print(f"{end_year}:  RMSE (CC, MAE, RMSE) = ({bias_cc_rmse:.6f}, {bias_mae_rmse:.6f}, {bias_rmse_rmse:.6f})")
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
    print(f"{end_year}:  (idx_corr={idx_corr}, idx_mae={idx_mae}, idx_rmse={idx_rmse}, idx_ce={idx_ce})\n")
#%%
cal_range = range(1986, 2026)
bias_history_mean = [res for res in bias_history if res['Year'] in cal_range]
bias_all = []
for n in range(len(bias_history_mean)):
    year = bias_history_mean[n]['Year']
    bias_cc_mae = bias_history_mean[n]['Bias_CC_mae']
    bias_mae_mae = bias_history_mean[n]['Bias_MAE_mae']
    bias_rmse_mae = bias_history_mean[n]['Bias_RMSE_mae']
    bias_cc_rmse = bias_history_mean[n]['Bias_CC_rmse']
    bias_mae_rmse = bias_history_mean[n]['Bias_MAE_rmse']
    bias_rmse_rmse = bias_history_mean[n]['Bias_RMSE_rmse']
    bias_all.append({
        'Year': int(year),
        'Bias_CC_mae': bias_cc_mae,
        'Bias_MAE_mae': bias_mae_mae,
        'Bias_RMSE_mae': bias_rmse_mae,
        'Bias_CC_rmse': bias_cc_rmse,
        'Bias_MAE_rmse': bias_mae_rmse,
        'Bias_RMSE_rmse': bias_rmse_rmse
    })
bias_all = pd.DataFrame(bias_all,index=cal_range)
mean_bias_cc_mae = bias_all['Bias_CC_mae'].mean()
mean_bias_mae_mae = bias_all['Bias_MAE_mae'].mean()
mean_bias_rmse_mae = bias_all['Bias_RMSE_mae'].mean()
mean_bias_cc_rmse = np.sqrt(bias_all['Bias_CC_rmse'].mean())
mean_bias_mae_rmse = np.sqrt(bias_all['Bias_MAE_rmse'].mean())
mean_bias_rmse_rmse = np.sqrt(bias_all['Bias_RMSE_rmse'].mean())
print("Mean Bias CC MAE:", mean_bias_cc_mae)
print("Mean Bias CC RMSE:", mean_bias_cc_rmse)
print('-----------------------')
print("Mean Bias MAE MAE:", mean_bias_mae_mae)
print("Mean Bias MAE RMSE:", mean_bias_mae_rmse)
print('-----------------------')
print("Mean Bias RMSE MAE:", mean_bias_rmse_mae)
print("Mean Bias RMSE RMSE:", mean_bias_rmse_rmse)
#%%
all_years_results_sel = [res for res in all_years_results if res['end_year'] in range(1986, 2026)]
bias_all = []
for n in range(len(all_years_results_sel)):
    year = all_years_results_sel[n]['end_year']
    pred_cc = df5.iloc[all_years_results_sel[n]['idx_corr'], get_year_index(index_sy,year)]
    true_cc = data_all.loc[year, data_all.columns[5]]
    bias_cc = np.abs(pred_cc - true_cc)
    print(year)
    print("Predicted CC :", f"{pred_cc:.2f}")
    print("Bias CC :", f"{bias_cc:.2f}")
    pred_mae = df5.iloc[all_years_results_sel[n]['idx_mae'], get_year_index(index_sy,year)]
    true_mae = data_all.loc[year, data_all.columns[5]]
    bias_mae = np.abs(pred_mae - true_mae)
    print("Predicted MAE :", f"{pred_mae:.2f}")
    print("Bias MAE :", f"{bias_mae:.2f}")
    pred_rmse = df5.iloc[all_years_results_sel[n]['idx_rmse'], get_year_index(index_sy,year)]
    true_rmse = data_all.loc[year, data_all.columns[5]]
    bias_rmse = np.abs(pred_rmse - true_rmse)
    print("Predicted RMSE :", f"{pred_rmse:.2f}")
    print("Bias RMSE :", f"{bias_rmse:.2f}")
#%%
rows = []
for res in all_years_results_sel:
    year = res['end_year']
    col_idx = get_year_index(index_sy, year)
    true_val = data_all.loc[year, data_all.columns[5]]
    pred_cc   = df5.iloc[res['idx_corr'], col_idx]
    pred_mae  = df5.iloc[res['idx_mae'],  col_idx]
    pred_rmse = df5.iloc[res['idx_rmse'], col_idx]
    rows.append({
        "Year": year,
        "True": round(true_val, 2),
        "Pred_CC": round(pred_cc, 2),
        "Bias_CC": round(abs(pred_cc - true_val), 2),
        "Pred_MAE": round(pred_mae, 2),
        "Bias_MAE": round(abs(pred_mae - true_val), 2),
        "Pred_RMSE": round(pred_rmse, 2),
        "Bias_RMSE": round(abs(pred_rmse - true_val), 2),
    })
bias_table = pd.DataFrame(rows)
bias_table
all_years_results_add_sel = [res for res in all_years_results if res['end_year'] in range(1986, 2026)]
bias_all_ce = []
for n in range(len(all_years_results_add_sel)):
    year = all_years_results_add_sel[n]['end_year']
    pred_ce = df5.iloc[all_years_results_add_sel[n]['idx_ce'], get_year_index(index_sy, year)]
    print(year)
    print("Predicted CE :", f"{pred_ce:.2f}")
    true_ce = data_all.loc[year, data_all.columns[5]]
    bias_ce = np.abs(pred_ce - true_ce)
    print(f"{year} CE Bias:", f"{bias_ce:.2f}")
    bias_all_ce.append({
        'Year': int(year),
        'Bias_CE': bias_ce
    })
bias_all_ce = pd.DataFrame(bias_all_ce)
mae_bias_ce = bias_all_ce['Bias_CE'].mean()
rmse_bias_ce = np.sqrt(np.power(bias_all_ce['Bias_CE'], 2).mean())
print("Mean Bias CE MAE:", mae_bias_ce)
print("Mean Bias CE RMSE:", rmse_bias_ce)
#%%
rows_ce = []
for res in all_years_results_add_sel:
    year = res['end_year']
    col_idx = get_year_index(index_sy, year)
    true_val = data_all.loc[year, data_all.columns[5]]
    pred_ce  = df5.iloc[res['idx_ce'], col_idx]
    rows_ce.append({
        "Year": year,
        "True": round(true_val, 2),
        "Pred_CE": round(pred_ce, 2),
        "Bias_CE": round(abs(pred_ce - true_val), 2),
    })
bias_table = pd.DataFrame(rows_ce)
bias_table
#%%
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
FONT_SIZE_TITLE = 13
FONT_SIZE_LABEL = 11
FONT_SIZE_TICK = 12
FONT_SIZE_LEGEND = 9
plt.rcParams.update({
    'axes.linewidth': 2,
    'font.size': FONT_SIZE_TICK,
    'xtick.major.width': 2,
    'ytick.major.width': 2,
})
sel_results = [res for res in all_years_results if res['end_year'] in range(2025, 2026)]
for n in range(len(sel_results)):
    data_sub = data_all.loc[sel_results[n]['end_year']-6:sel_results[n]['end_year']]
    df5_s_sub = df5.loc[:, sel_results[n]['end_year']-6:sel_results[n]['end_year']]
    idx_ce = sel_results[n]['idx_ce']
    end_year = sel_results[n]['end_year']
    df5_sc2 = df5_s_sub.loc[df5_sc.index, :]
    fig, ax = plt.subplots(1, 1, figsize=(8, 3.4), dpi=300, facecolor='#FFFFFF')
    color_real = "#3E62AD"
    color_pred = "#FF8A00"
    color_gray = "#9c9a9a"
    color_dark = "#363635"
    alpha_base = 0.6
    ax.axvspan(data_sub.index[0], data_sub.index[-2], 
               facecolor='#8ad1f5', alpha=0.15, zorder=-2)
    for i in range(df5_sc.shape[0]):
        ax.plot(data_sub.index[:-1], df5_sc2.iloc[i, :-1], 
                color=color_gray, linewidth=1, alpha=0.15, zorder=1, 
                label='_nolegend_')
    ax.plot(data_sub.index[:-1], data_sub.iloc[:-1, pf], 
            color=color_real, linewidth=3.5, marker='s', markersize=8,
            alpha=alpha_base, label='Real' if n == 0 else "", zorder=3)
    ax.plot(data_sub.index[-2:], data_sub.iloc[-2:, pf], 
            color=color_real, linewidth=3.5, linestyle='--', marker='s',
            markersize=8, alpha=alpha_base, zorder=3)
    ax.plot(data_sub.index[0:1], df5_s_sub.iloc[idx_ce, 0:1], 
            color=color_dark, linewidth=2, marker='x', markersize=8,
            alpha=0.8, label='_nolegend_', zorder=4)
    ax.plot(data_sub.index[1:-1], df5_s_sub.iloc[idx_ce, 1:-1], 
            color=color_pred, linewidth=3.5, marker='o', markersize=8,
            alpha=alpha_base, label='Selected' if n == 0 else "", zorder=4)
    ax.plot(data_sub.index[-1:], df5_s_sub.iloc[idx_ce, -1:], 
            color=color_pred, linewidth=0,  
            marker='o', markersize=8,       
            markeredgecolor='black', 
            markeredgewidth=1.5,
            markerfacecolor=color_pred,      
            alpha=alpha_base, zorder=5,      
            label='Forecast' if n == 0 else "")
    ax.plot(data_sub.index[-2:], df5_s_sub.iloc[idx_ce, -2:], 
            color=color_pred, linewidth=2.5, linestyle='--',
            marker='o', markersize=8,       
            markerfacecolor='none',           
            markeredgecolor='none',           
            alpha=alpha_base, zorder=3)       
    ax.axvline(x=end_year, color='k', linestyle='--', linewidth=1.2, 
               alpha=0.6, zorder=1)
    ax.set_title(f'{end_year}', loc='left', 
                 fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=8)
    ax.set_ylim(21, 37)
    ax.set_xlim(data_sub.index[0]-0.5, data_sub.index[-1]+0.5)
    ax.set_xticks(data_sub.index[::1])
    ax.set_xticklabels(data_sub.index[::1], fontsize=FONT_SIZE_TICK)
    ax.set_ylabel('Pentad', fontsize=FONT_SIZE_LABEL)
    ax.yaxis.set_label_coords(-0.05, 0.5)
    ax.spines['top'].set_visible(True)
    ax.spines['top'].set_linewidth(2.0)
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_linewidth(2)
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_linewidth(2)
    ax.spines['right'].set_color('black')
    ax.grid(alpha=0.25, ls='--', axis='x', zorder=0)
    secax = ax.secondary_xaxis('bottom')
    secax.set_xticks(data_sub.index)
    rel_labels = [f'(t-{6-i})' if i < 6 else '(t)' 
                  for i in range(len(data_sub.index))]
    secax.set_xticklabels(rel_labels, fontsize=FONT_SIZE_TICK, fontweight='bold')
    secax.spines['bottom'].set_position(('outward', 18))
    secax.spines['bottom'].set_visible(False)
    secax.tick_params(axis='x', length=0, pad=2)
    for i, label in enumerate(secax.get_xticklabels()):
        if i < 6:
            label.set_color("#1a1a1a")
        else:
            label.set_color("#C41E3A")
    legend_elements = [
        Line2D([0], [0], color=color_real, lw=2.5, marker='s', markersize=6,
               alpha=alpha_base, linestyle='-', label='Obs'),
        Line2D([0], [0], color=color_pred, lw=2.5, marker='o', markersize=6,
               alpha=alpha_base, linestyle='-', label='Sel'),
        Line2D([0], [0], color=color_pred, lw=2, marker='o', markersize=6,
               markeredgecolor='black', linestyle='--', label='Pre'),
        Line2D([0], [0], color=color_gray, lw=1, alpha=0.3, linestyle='-',
               label='Top 10%'),
        Line2D([0], [0], color=color_dark, lw=0, marker='x', markersize=8,
               markeredgewidth=2, linestyle='None', label='Exc')
    ]
    ax.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.63, 0.98), ncol=5, fontsize=FONT_SIZE_LEGEND,
              frameon=False, columnspacing=1.2)
    plt.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.22)
    plt.savefig(f'../features_separate/prediction_{end_year}.svg', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(":", df5_s_sub.iloc[idx_ce, :])
#%%
data_all_2 = data_all.loc['1979':]
data_all_ss = data_all_2.apply(lambda x:(x-np.mean(x))/np.std(x)).values  
stt = data_all_ss[:,0]
enso = data_all_ss[:,1]
sam = data_all_ss[:,2]
y_ss = 27.98 - 0.48*stt - 0.66*enso - 1.03*sam
y_ss = pd.DataFrame(y_ss, index=data_all_2.index, columns=['y_ss'])
from matplotlib.lines import Line2D
plt.rcParams['font.sans-serif'] = ['Calibri']  
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['font.size'] = 14
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
fig, ax = plt.subplots(figsize=(12, 3.5), dpi=300, facecolor='white')
N = df5.shape[0]
df5.columns = list(range(1979, 2026))
df5g = df5
for i in range(int(N)):
    ax.plot(df5g.columns, df5g.iloc[i, :], color="#415DA5", alpha=0.1,label='_nolegend_')
line1 = ax.plot(df5g.columns, data_all_2.iloc[:, -1], label='OBS', color="#131313", lw=3,ls = '-',alpha=0.9)[0]
ax.axvspan(1979, 2025-n, color="#E9E0D6", alpha=0.3)
ax.axvspan(2016, 2025, color="#4396DF", alpha=0.3)
ax.grid(alpha=0.2,ls = '--',zorder=0,color="#444040")
ax.set_ylim(20, 38)
ax.set_xlim(1978, 2026)
ax.set_xticks(np.arange(1979, 2026, 5))
ax.set_yticks(np.arange(20, 39, 1))
yticks = np.arange(20, 39, 1)
yticklabels = [str(y) if y % 2 == 0 else '' for y in yticks]
ax.set_yticklabels(yticklabels)
ax.set_ylabel('Onset Pentad', fontsize=16)
ax.set_xlabel('Years', fontsize=16)
ax.set_title('Generated Models', fontsize=20,fontweight='bold',pad=8)
ax.text(1998.5, 21, f'Train Model', fontsize=16, color='black', ha='center')
ax.text(2019.5, 21, f'Test Model', fontsize=16, color='black', ha='center')
ml_model_legend = Line2D([0], [0], color="#415DA5", lw=2, linestyle='-', label='Multi-Model',alpha=0.6)
handles = [line1, ml_model_legend]
labels = [h.get_label() for h in handles]
ax.legend(handles, labels, fontsize=10,loc='upper right',ncol=3,columnspacing=0.8)  
plt.subplots_adjust(left=0.07, right=0.85, top=0.85, bottom=0.2)
plt.show()
