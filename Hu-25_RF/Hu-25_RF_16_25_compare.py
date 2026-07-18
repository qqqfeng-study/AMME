#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error 
from scipy.stats import pearsonr
import os
import shap
import joblib
from joblib import Parallel, delayed, dump
from tqdm import tqdm
data_all = pd.read_csv('./precursor_factors.csv',index_col=0)
data_all = data_all.drop(columns=['ao_mar','lst','vm_eof2','mse_850','tp_feb'])
end_year = 1999
data_all = data_all.loc['1979':end_year]
data_all_ss = data_all.apply(lambda x:(x-np.mean(x))/np.std(x)).values  
#%%
la = len(data_all)
nn = la-10
test_x_79_14 = data_all_ss[0:nn,0:3]
test_y = data_all.iloc[:, 3].values
test_y_79_14 = test_y[0:nn] 
#%%
if not os.path.exists(f'./model_rf_{end_year}'):
    os.makedirs(f'./model_rf_{end_year}')
    print("model_rf")
else:
    print("model_rf")
    pass
#%%
x_test_fixed = data_all_ss[nn:la, 0:3]
y_test_fixed = data_all.iloc[nn:la, 3].values
#%%
def train_and_save(i):
    RF = RandomForestRegressor(
        n_estimators=100,
        criterion='absolute_error',
        n_jobs=-1  
    )
    x_train, x_test, y_train, y_test = train_test_split(
        test_x_79_14, test_y_79_14, test_size=0.2
    )
    rf = RF.fit(x_train, y_train.ravel())
    val = rf.score(x_test, y_test)
    test = rf.score(x_test_fixed, y_test_fixed)
    dump(rf, f'./model_rf_{end_year}/bst_model_{i+1}.pkl')
    return val, test
N = 10000
results = Parallel(n_jobs=-1)(
    delayed(train_and_save)(i) for i in tqdm(range(N), desc="", unit="")
)
val_score, test_score = map(np.array, zip(*results))
print(f": {val_score.mean():.4f}")
print(f": {test_score.mean():.4f}")
#%%
def load_and_predict(kk):
    fn_in = f'./model_rf_{end_year}/bst_model_{kk+1}.pkl'
    rf_load = joblib.load(fn_in)
    return rf_load.predict(data_all_ss[:, 0:3])
y_pred_all = Parallel(n_jobs=-1)(
    delayed(load_and_predict)(kk) for kk in tqdm(range(N), desc="", unit="")
)
y_pred_all = np.array(y_pred_all)
import pandas as pd
df = pd.DataFrame(y_pred_all)
excel_filename = f'./yy_pre_rf_model_{end_year}.csv'
df.to_csv(excel_filename, index=False)
#%%
excel_filename = f'./yy_pre_rf_model_{end_year}.csv'
df = pd.read_csv(excel_filename)
N = 10000
tr = nn 
py = la
pf = data_all.shape[1] - 1 
mapping_offsets = {
    2025: (0,  0, -1),   
    2024: (0,  0, -1),  
    2023: (0,  0, -1),  
    2022: (0,  0, -1),  
    2021: (0,  0, -1),  
    2020: (0,  0, -1),  
    2019: (0,  0, -1),  
    2018: (0,  0, -1),  
    2017: (0,  0, -1),  
    2016: (0,  0, -1),  
    2015: (0,  0, -1),  
    2014: (0,  0, -1),  
    2013: (0,  0, -1),  
    2012: (0,  0, -1),  
    2011: (0,  0, -1),  
    2010: (0,  0, -1),  
    2009: (0,  0, -1),  
    2008: (0,  0, -1),  
    2007: (0,  0, -1),  
    2006: (0,  0, -1),  
    2005: (0,  0, -1),  
    2004: (0,  0, -1),  
    2003: (0,  0, -1),  
    2002: (0,  0, -1),  
    2001: (0,  0, -1),  
    2000: (0,  0, -1),  
}
def get_values_by_year(year, tr, py, mapping=mapping_offsets):
    if year not in mapping:
        raise KeyError(f"Year {year} not configured")
    off1, off2, off3 = mapping[year]
    val1 = int(tr + off1)
    val2 = int(tr + off2)
    val3 = int(py + off3)
    return val1, val2, val3
get_year = end_year
v1, v2, v3 = get_values_by_year(get_year, tr, py)
train_a, train_b = 0, v1   
test_a, test_b = v2, v3    
corr_train, corr_test = [], []
mae_train, mae_test = [], []
rmse_train, rmse_test = [], []
for i in range(N):
    ct = pearsonr(data_all.iloc[train_a:train_b, pf], df.iloc[i, train_a:train_b])[0]
    ce = pearsonr(data_all.iloc[test_a:test_b, pf], df.iloc[i, test_a:test_b])[0]
    corr_train.append(ct)
    corr_test.append(ce)
    mt = mean_absolute_error(data_all.iloc[train_a:train_b, pf], df.iloc[i, train_a:train_b])
    me = mean_absolute_error(data_all.iloc[test_a:test_b, pf], df.iloc[i, test_a:test_b])
    mae_train.append(mt)
    mae_test.append(me)
    rt = np.sqrt(mean_squared_error(data_all.iloc[train_a:train_b, pf], df.iloc[i, train_a:train_b]))
    re = np.sqrt(mean_squared_error(data_all.iloc[test_a:test_b, pf], df.iloc[i, test_a:test_b]))
    rmse_train.append(rt)
    rmse_test.append(re)
#%%
result_df = pd.DataFrame({
    'corr_train': corr_train,
    'corr_test': corr_test,
    'mae_train': mae_train,
    'mae_test': mae_test,
    'rmse_train': rmse_train,
    'rmse_test': rmse_test,
})
max_corr_index = np.argmax(result_df['corr_test'])
min_mae_index = np.argmin(result_df['mae_test'])
min_rmse_index = np.argmin(result_df['rmse_test'])
print("", max_corr_index)
print("MAE", min_mae_index)
print("RMSE", min_rmse_index)
corr_test_best = result_df['corr_test'].loc[max_corr_index]
mae_test_best = result_df['mae_test'].loc[min_mae_index]
rmse_test_best = result_df['rmse_test'].loc[min_rmse_index]
corr_9 = np.corrcoef(data_all.iloc[nn:py-1, 3], df.iloc[max_corr_index, nn:py-1])[0, 1]
print("9", corr_9)
corr_before = np.corrcoef(data_all.iloc[:nn, 3], df.iloc[max_corr_index, :nn])[0, 1]
print("37", corr_before)
#%%
mae_9 = mean_absolute_error(data_all.iloc[nn:py-1, 3], df.iloc[min_mae_index, nn:py-1])
print("9MAE", mae_9)
mae_before = mean_absolute_error(data_all.iloc[:nn, 3], df.iloc[min_mae_index, :nn])
print("37MAE", mae_before)
#%%
rmse_9 = np.sqrt(mean_squared_error(data_all.iloc[nn:py-1, 3], df.iloc[min_rmse_index, nn:py-1]))
print("9RMSE", rmse_9)
rmse_before = np.sqrt(mean_squared_error(data_all.iloc[:nn, 3], df.iloc[min_rmse_index, :nn]))
print("37RMSE", rmse_before)
#%%
df.columns = np.arange(1979, get_year + 1)
print(f"{get_year}", df.iloc[max_corr_index].loc[get_year])
print(f"MAE{get_year}", df.iloc[min_mae_index].loc[get_year])
print(f"RMSE{get_year}", df.iloc[min_rmse_index].loc[get_year])
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.lines import Line2D
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['font.size'] = 14
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
year_start = 1979
fig, ax = plt.subplots(figsize=(15, 4))
data_all.iloc[:, 3].plot(ax=ax, label='Observation', marker='o', color='k', lw=2)
df.iloc[max_corr_index, :].plot(ax=ax, label='MM-25', color="#F15348", marker='s', lw=2)
ax.axvspan(1940, 2025-17, color="#777573", alpha=0.3)
ax.axvspan(2025-17, 2025-1, color="#035FAF", alpha=0.3)
ax.axvspan(2025-1, 2025, color="#F89C9C", alpha=0.5)
ax.grid()
ax.set_ylim(22, 36)
ax.set_xlim(1976, 2026)
ax.set_xticks(np.arange(1976, 2026, 5))
ax.set_yticks(np.arange(22, 37, 1))
yticks = np.arange(22, 37, 1)
yticklabels = [str(y) if y % 2 == 0 else '' for y in yticks]
ax.set_yticklabels(yticklabels)
ax.set_xlabel('Year', fontsize=16)
ax.set_ylabel('Onset Pentad', fontsize=16)
ax.set_title('(a) Selected base on the best correlation coefficient', fontsize=18)
ax.text(1990, 34.5, f'CORR: {corr_before:.2f}', fontsize=14, color='black', ha='center')
ax.text(2019.5, 34.5, f'CORR: {corr_9:.2f}', fontsize=14, color='black', ha='center')
ax.legend(['Observation', 'MM-25'], loc='lower right')
plt.show()
#%%
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['font.size'] = 14
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
year_start = 1979
fig, ax = plt.subplots(figsize=(16, 4))
for i in range(1000):
    df.iloc[i, :].plot(color="#3619F3", alpha=0.1,label="_nolegend_")
data_all.iloc[:, 3].plot(ax=ax, label='Observation', marker='o', color='k', lw=2)
df.iloc[max_corr_index, :].plot(ax=ax, label='MM-25', color="#F18E48", marker='s', lw=2,ls = '-')
line1 = ax.lines[-2]
line2 = ax.lines[-1]
ax.axvspan(1979, year_start+v1, color="#777573", alpha=0.3)
ax.axvspan(year_start+v1, year_start+v3-1, color="#035FAF", alpha=0.3)
ax.axvspan(year_start+v3-1, year_start+v3, color="#F89C9C", alpha=0.5)
ax.grid()
ax.set_ylim(22, 36)
ax.set_xlim(1978, 2026)
ax.set_xticks(np.arange(1979, 2026, 5))
ax.set_yticks(np.arange(22, 37, 1))
yticks = np.arange(22, 37, 1)
yticklabels = [str(y) if y % 2 == 0 else '' for y in yticks]
ax.set_yticklabels(yticklabels)
ax.set_ylabel('Onset Pentad', fontsize=16)
ax.set_title('(b) Model Selected by the Best Mean Absolute Error', fontsize=18)
ax.text(1990, 34.5, f'MAE: {mae_before:.2f}', fontsize=14, color='black', ha='center')
ax.text(2019.5, 34.5, f'MAE: {mae_9:.2f}', fontsize=14, color='black', ha='center')
ml_model_legend = Line2D([0], [0], color="#3619F3", lw=2, linestyle='-', label='ML_model')
handles = [line1, line2, ml_model_legend]
labels = [h.get_label() for h in handles]
ax.legend(handles, labels, loc='best', fontsize=12,bbox_to_anchor=(1, 0.6))
plt.show()
