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
from matplotlib.lines import Line2D
import math
def get_year_index(index, year):
    return index.year.get_loc(year)
def get_eva_for_model_fixed_window(data_sub, df5_sc, allowed_lens, model_idx):
    x1 = data_sub.iloc[:allowed_lens].values.astype(float)
    y1 = df5_sc.iloc[model_idx, :allowed_lens].values.astype(float)
    if x1.size < 2 or np.allclose(x1, x1[0]) or np.allclose(y1, y1[0]):
        corr_r2 = np.nan
    else:
        corr_raw, _ = pearsonr(x1, y1)
        corr_r2 = float(corr_raw)
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
df5 = pd.read_csv(f'../final/migrated_500000.csv')
df5.columns = list(range(1979, 2026))
data_all = pd.read_csv('../precursor_factors.csv',index_col=0)
data_all = data_all.drop(columns=['mse_850','vm_eof2','tp_feb'])
range_start = 1985
range_end = 2026
index_sy = pd.date_range(start='1979', end=f'{range_end+1}', freq='YS')
bias_history = []  
all_years_results = []
df5
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
        print("mae:", errors.mean())
        p = 10
        th = np.percentile(errors, p)
        close_models_2 = np.where(errors <= th)[0]
        if close_models_2.size == 0:
            print(f"{end_year}: {p} (={th:.4f})")
        else:
            print(f"{end_year}: {p}={th:.4f} {len(close_models_2)} ")
        df5_sc = df5_sg.iloc[close_models_2, :]
    results = Parallel(n_jobs=24)(delayed(get_eva_for_model_fixed_window)(data_sg, df5_sc, allowed_lens, i) for i in range(len(df5_sc)))
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
    prev_biases = sorted(all_prev, key=lambda x: x['Year'], reverse=True)[:6]
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
        'sel_model_idx': close_models_2.tolist(),
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
cal_range = range(range_start+1, range_end)
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
all_years_results_sel = [res for res in all_years_results if res['end_year'] in range(2016, range_end)]
metric_all = []
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
    metric_all.append({
        'Year': int(year),
        'True': true_cc,
        'Predicted_CC': pred_cc,
        'Predicted_MAE': pred_mae,
        'Predicted_RMSE': pred_rmse,
    })
metric_all = pd.DataFrame(metric_all)
cc_r = pearsonr(metric_all['True'], metric_all['Predicted_CC'])[0]
cc_mae = mean_absolute_error(metric_all['True'], metric_all['Predicted_CC'])
cc_rmse = np.sqrt(mean_squared_error(metric_all['True'], metric_all['Predicted_CC']))
mae_r = pearsonr(metric_all['True'], metric_all['Predicted_MAE'])[0]
mae_mae = mean_absolute_error(metric_all['True'], metric_all['Predicted_MAE'])
mae_rmse = np.sqrt(mean_squared_error(metric_all['True'], metric_all['Predicted_MAE']))
rmse_r = pearsonr(metric_all['True'], metric_all['Predicted_RMSE'])[0]
rmse_mae = mean_absolute_error(metric_all['True'], metric_all['Predicted_RMSE'])
rmse_rmse = np.sqrt(mean_squared_error(metric_all['True'], metric_all['Predicted_RMSE']))
print("CC R:", cc_r)
print("CC MAE:", cc_mae)
print("CC RMSE:", cc_rmse)
print('-----------------------')
print("MAE R:", mae_r)
print("MAE MAE:", mae_mae)
print("MAE RMSE:", mae_rmse)
print('-----------------------')
print("RMSE R:", rmse_r)
print("RMSE MAE:", rmse_mae)
print("RMSE RMSE:", rmse_rmse)
print('-----------------------')
all_years_results_add_sel = [res for res in all_years_results if res['end_year'] in range(2016, 2026)]
bias_all_ce = []
for n in range(len(all_years_results_add_sel)):
    year = all_years_results_add_sel[n]['end_year']
    pred_ce = df5.iloc[all_years_results_add_sel[n]['idx_ce'], get_year_index(index_sy, year)]
    true_ce = data_all.loc[year, data_all.columns[5]]
    bias_ce = np.abs(pred_ce - true_ce)
    print(year)
    print("Predicted CE :", f"{pred_ce:.2f}")
    print(f"{year} CE Bias:", f"{bias_ce:.2f}")
    bias_all_ce.append({
        'Year': int(year),
        'Bias_CE': bias_ce,
        'Predicted_CE': pred_ce,
        'True_CE': true_ce
    })
bias_all_ce = pd.DataFrame(bias_all_ce)
ce_r = pearsonr(bias_all_ce['True_CE'], bias_all_ce['Predicted_CE'])[0]
mae_bias_ce = bias_all_ce['Bias_CE'].mean()
rmse_bias_ce = np.sqrt(np.power(bias_all_ce['Bias_CE'], 2).mean())
print("CE R:", ce_r)
print("Mean Bias CE MAE:", mae_bias_ce)
print("Mean Bias CE RMSE:", rmse_bias_ce)
all_years_results_add_sel = [res for res in all_years_results if res['end_year'] in range(2016, 2026)]
real_list = []
pred_list = []
set_list = []
years_idx_list = []
year_labels = []
for n in range(len(all_years_results_add_sel)):
    end_year = all_years_results_add_sel[n]['end_year']
    data_sub = data_all.loc[end_year-6:end_year]
    df5_s_sub = df5.loc[:, end_year-6:end_year]
    df5_sc2 = df5_s_sub.loc[all_years_results_add_sel[n]['sel_model_idx'], :]
    idx_ncorr = all_years_results_add_sel[n]['idx_ce']
    real = data_sub.iloc[:, pf].values  
    pred = df5_s_sub.iloc[idx_ncorr, :].values
    real_list.append(real)
    pred_list.append(pred)
    set_list.append(df5_sc2)
    years_idx_list.append(data_sub.index)   
    year_labels.append(str(end_year))
n_panels = len(real_list)
fig, axes = plt.subplots(n_panels, 1, figsize=(8, 0.9 * n_panels), dpi=300, sharex=True)  
if n_panels == 1:
    axes = [axes]
for i, ax in enumerate(axes):
    years_idx = years_idx_list[i]
    real = real_list[i]
    pred = pred_list[i]
    df5_sc2 = set_list[i]
    x = np.array(years_idx)
    y_real = np.array(real)
    y_pred = np.array(pred)
    end_year_panel = int(year_labels[i])
    special_years = []
    if end_year_panel == 2024:
        special_years = [2018, 2019]
    elif end_year_panel == 2025:
        special_years = [2019]
    if len(special_years) > 0:
        years_num = x.astype(int)
        mask_special_pre = np.isin(years_num, special_years)
    else:
        mask_special_pre = np.zeros_like(x, dtype=bool)
    mask_normal_pre = ~mask_special_pre
    alpha_mbase = 0.6
    alpha_lbase = 0.6
    color_real = "#3E62AD"
    color_pred = "#FF8A00"
    color_special = "#777777"   
    for m in range(df5_sc2.shape[0]):
        ax.plot(x[:allowed_lens+2], df5_sc2.iloc[m, :], color="#9c9a9a", linewidth=1, alpha=0.1,zorder=1, label='_nolegend_')
    ax.plot(x[-1:], y_real[-1:], linestyle='None', marker='s', color=color_real,
            alpha=alpha_mbase, markersize=7, label='')
    if len(x) >= 2:
        if len(x) == 2:
            ax.plot(x[-2:], y_real[-2:], linestyle='--', color=color_real, lw=3,
                    alpha=alpha_lbase, marker='o')
        else:
            ax.plot(x[:-1], y_real[:-1], linestyle='-', color=color_real, lw=3,
                    alpha=alpha_lbase, label='Real' if i == 0 else "", marker='s')
            ax.plot(x[-2:], y_real[-2:], linestyle='--', color=color_real, lw=2,
                    alpha=alpha_lbase)
    ax.plot(x[-1:], y_pred[-1:], linestyle='None', marker='o', color=color_pred,markeredgecolor='black',alpha=alpha_mbase, markersize=7, label='')
    if len(x) >= 2:
        x_n = x[mask_normal_pre]
        y_pred_n = y_pred[mask_normal_pre]
        if len(x_n) >= 2:
            years_n = x_n.astype(int)
            segments = []
            start_idx = 0
            for k in range(1, len(x_n)):
                if years_n[k] != years_n[k-1] + 1:
                    segments.append((start_idx, k))
                    start_idx = k
            segments.append((start_idx, len(x_n)))
            for (s, e) in segments:
                xs = x_n[s:e]
                ys = y_pred_n[s:e]
                if len(xs) == 1:
                    continue
                ax.plot(xs[:-1], ys[:-1], linestyle='-', color=color_pred, lw=3,
                        alpha=alpha_lbase,
                        label='Select' if (i == 0 and s == 0) else "", marker='o')
                ax.plot(xs[-2:], ys[-2:], linestyle='--', color=color_pred, lw=2,
                        alpha=alpha_lbase)
        if mask_special_pre.any():
            ax.plot(x[mask_special_pre], y_pred[mask_special_pre],
                    linestyle='None', marker='x', color=color_special,
                    markersize=7, mew=2)
    ax.axvspan(x[0], x[-1]-1, ymin=0, ymax=1, facecolor='#8ad1f5', alpha=0.2, zorder=-1)
    ax.text(0.01, 0.18, f"{year_labels[i]}", transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='left', color='#333333')
    ax.set_ylim(22, 37)
    if i < n_panels - 1:
        ax.tick_params(
            axis='x',
            which='both',
            bottom=False,      
            labelbottom=False  
        )
    ax.spines['top'].set_visible(False)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['top'].set_color('#dddddd')
    ax.spines['top'].set_zorder(0)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_linewidth(2)
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_visible(True)
    ax.spines['right'].set_linewidth(2)
    ax.spines['right'].set_color('black')
    if i == 0:
        ax.spines['top'].set_visible(True)
        ax.spines['top'].set_linewidth(2.0)
        ax.spines['top'].set_color('black')
    if i < n_panels - 1:
        ax.spines['bottom'].set_visible(False)
    else:
        ax.spines['bottom'].set_linewidth(2.0)
        ax.spines['bottom'].set_color('black')
    ax.spines['right'].set_visible(True)
    ax.grid(alpha=0.25, ls='--', axis='x')
    ax.set_xlim(years_idx_list[0].min(), years_idx_list[-1].max()+1)
    ax.set_xticks(np.arange(years_idx_list[0].min()-1, years_idx_list[-1].max()+2, 2))
    ax.set_xticklabels(np.arange(years_idx_list[0].min()-1, years_idx_list[-1].max()+2, 2))
    ax.tick_params(axis='both', which='major', labelsize=10, length=4, width=2)
plt.subplots_adjust(hspace=0.2, top=0.98, bottom=0.06, left=0.12, right=0.98)
axes[-1].set_xlabel('Years', fontsize=12)
legend_elements = [
    Line2D(
        [0], [0],
        color="#3E62AD",
        lw=1.5,
        marker='s',
        markersize=7,
        alpha=0.6,
        linestyle='-',
        label='Obs'
    ),
    Line2D(
        [0], [0],
        color="#FF8A00",
        lw=1.5,
        marker='o',
        alpha=0.6,
        markersize=7,
        linestyle='-',
        label='Sel'
    ),
    Line2D(
        [0], [0],
        color='#FF8A00',
        lw=1.5,
        marker='o',
        alpha=0.6,
        markersize=7,
        markeredgecolor='black',
        linestyle='--',
        label='Pre'
    ),
    Line2D(
        [0], [0],
        color="#9c9a9a",
        lw=1,
        marker='None',
        alpha=0.5,
        linestyle='-',
        label='Set'
    ),
    Line2D(
        [0], [0],
        color="#777777",
        lw=0,
        marker='x',
        markersize=8,
        markeredgewidth=2,
        linestyle='None',
        label='Exc'
    )
]
fig.legend(
    handles=legend_elements,
    loc='upper left',
    bbox_to_anchor=(0.48, 0.98),
    frameon=False,
    fontsize=9,
    ncol=5
)
fig.text(0.05, 0.5, 'Pentad', va='center', rotation='vertical', fontsize=12)
plt.show()
