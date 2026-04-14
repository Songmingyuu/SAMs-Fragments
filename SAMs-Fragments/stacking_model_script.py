import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, LeaveOneOut, RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
    StackingRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import (
    RidgeCV,
    Lasso,
    ElasticNet,
    BayesianRidge
)
from sklearn.experimental import enable_hist_gradient_boosting  
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')  


file_path = r'your file'
df = pd.read_excel(file_path, dtype=str)


df['Anchoring Group'] = df['Anchoring Group'].apply(lambda x: list(map(int, x)))
df['Linker Group'] = df['Linker Group'].apply(lambda x: list(map(int, x)))
df['Functional Head'] = df['Functional Head'].apply(lambda x: list(map(int, x)))


df['Full Code'] = df['Anchoring Group'] + df['Linker Group'] + df['Functional Head']


y = pd.to_numeric(df['PCE/%'], errors='coerce')


x = df['Full Code']
x = x[y.notna()]
y = y.dropna()


x = np.array(x.tolist())


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.n, random_state=24)


scaler = StandardScaler()
x = scaler.fit_transform(x)


param_distributions = {
    'gbr': {
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'subsample': [0.6, 0.8, 1.0],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    },
    'rf': {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    },
    'svr': {
        'C': [0.1, 1, 10, 100, 1000, 10000],
        'epsilon': [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
        'gamma': ['scale', 'auto']
    },
    'ridge': {
        'alphas': [0.1, 1.0, 10.0, 100.0]
    },
    'lasso': {
        'alpha': [0.01, 0.1, 1.0, 10.0],
        'max_iter': [1000, 2000, 3000]
    },
    'knn': {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    },
    'dt': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    },
    'xgb': {
        'n_estimators': [100, 200, 300, 400],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.01, 0.1],
        'reg_lambda': [1, 1.5, 2]
    },
    'mlp':  {
        'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['adam', 'lbfgs', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'invscaling', 'adaptive']
    }    
}

models = {
    'gbr': GradientBoostingRegressor(random_state=24),
    'rf': RandomForestRegressor(random_state=24),
    'svr': SVR(),
    'ridge': RidgeCV(),
    'lasso': Lasso(random_state=24),
    'knn': KNeighborsRegressor(),
    'dt': DecisionTreeRegressor(random_state=24),
    'xgb': xgb.XGBRegressor(random_state=24, objective='reg:squarederror'),
    'mlp': MLPRegressor(random_state=24, max_iter=1000),
}

def tune_model(model_name, model, param_dist, X, y, cv=kf, n_iter=n, random_state=24):
    
    print(f"\n正在优化模型: {model_name}")
    
    
    if model_name in ['ridge', 'lasso']:
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=len(param_dist['alphas']) if model_name == 'ridge' else len(param_dist['alpha']),
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=random_state
        )
    else:
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=random_state
        )
    
    search.fit(X, y)
    best_estimator = search.best_estimator_
    best_params = search.best_params_
    
    print(f"最佳参数 for {model_name}: {best_params}")
    return best_estimator, best_params


kf = KFold(n_splits=n, shuffle=True, random_state=24)


best_models = {}
best_params = {}

for name, model in models.items():
    tuned_model, params = tune_model(name, model, param_distributions[name], x, y, cv=kf, n_iter=n, random_state=24)
    best_models[name] = tuned_model
    best_params[name] = params


base_models = [
    ('gbr', best_models['gbr']),
    ('rf', best_models['rf']),
    ('svr', best_models['svr']),
    ('ridge', best_models['ridge']),
    ('lasso', best_models['lasso']),
    ('knn', best_models['knn']),
    ('dt', best_models['dt']),
    ('xgb', best_models['xgb']),
    ('mlp', best_models['mlp'])
]


model_pairs = list(combinations(base_models, 2))
print(f"\n共有 {len(model_pairs)} 种模型组合。")

def calculate_accuracy(y_true, y_pred, tolerance=0.1):
    
    return np.mean(np.abs(y_true - y_pred) / y_true <= tolerance)

def mean_absolute_percentage_error(y_true, y_pred):
    
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_stacking(pair, meta_model, X_train, y_train, X_test, y_test, cv):
    
    stacking_regressor = StackingRegressor(
        estimators=list(pair),
        final_estimator=meta_model,
        cv=cv,
        passthrough=False,
        n_jobs=-1
    )
    
    stacking_regressor.fit(X_train, y_train)
    y_pred_test = stacking_regressor.predict(X_test)
    y_pred_train = stacking_regressor.predict(X_train)
    
    
    mse_test = mean_squared_error(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    r_test, _ = pearsonr(y_test, y_pred_test)
    accuracy_test = calculate_accuracy(y_test, y_pred_test, tolerance=0.1)
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
    
    
    mse_train = mean_squared_error(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    r_train, _ = pearsonr(y_train, y_pred_train)
    accuracy_train = calculate_accuracy(y_train, y_pred_train, tolerance=0.1)
    mape_train = mean_absolute_percentage_error(y_train, y_pred_train)
    
    metrics = {
        'Test_R2': r2_test,
        'Test_MSE': mse_test,
        'Test_MAE': mae_test,
        'Test_R': r_test,
        'Test_Accuracy': accuracy_test,
        'Test_MAPE': mape_test,
        'Train_R2': r2_train,
        'Train_MSE': mse_train,
        'Train_MAE': mae_train,
        'Train_R': r_train,
        'Train_Accuracy': accuracy_train,
        'Train_MAPE': mape_train
    }
    
    return metrics

num = random.choice(range(0,100))

num_iterations = num  
meta_model = RidgeCV()  


all_results = {f"{pair[0][0]} + {pair[1][0]}": {
    'Test_R2': [], 'Test_MSE': [], 'Test_MAE': [], 'Test_R': [], 'Test_Accuracy': [], 'Test_MAPE': [],
    'Train_R2': [], 'Train_MSE': [], 'Train_MAE': [], 'Train_R': [], 'Train_Accuracy': [], 'Train_MAPE': []
} for pair in model_pairs}

for iteration in range(num_iterations):
    print(f"\n=== 迭代次数: {iteration + 1} / {num_iterations} ===")
    
    X_train_iter, X_test_iter, y_train_iter, y_test_iter = train_test_split(
        x, y, test_size=0.2, random_state=24
    )
    
    for pair in model_pairs:
        pair_names = f"{pair[0][0]} + {pair[1][0]}"
        print(f"\n评估模型组合：{pair_names}")
        
        
        metrics = evaluate_stacking(pair, meta_model, X_train_iter, y_train_iter, X_test_iter, y_test_iter, kf)
        
        
        print(f"测试集 MSE: {metrics['Test_MSE']:.4f}")
        print(f"测试集 MAE: {metrics['Test_MAE']:.4f}")
        print(f"测试集 R²: {metrics['Test_R2']:.4f}")
        print(f"测试集 R: {metrics['Test_R']:.4f}")
        print(f"测试集 Accuracy (±10%): {metrics['Test_Accuracy']*100:.2f}%")
        print(f"测试集 MAPE: {metrics['Test_MAPE']:.2f}%")
        
        print(f"训练集 MSE: {metrics['Train_MSE']:.4f}")
        print(f"训练集 MAE: {metrics['Train_MAE']:.4f}")
        print(f"训练集 R²: {metrics['Train_R2']:.4f}")
        print(f"训练集 R: {metrics['Train_R']:.4f}")
        print(f"训练集 Accuracy (±10%): {metrics['Train_Accuracy']*100:.2f}%")
        print(f"训练集 MAPE: {metrics['Train_MAPE']:.2f}%")
        
        
        for key in metrics:
            all_results[pair_names][key].append(metrics[key])


summary_results = []
for pair, metrics in all_results.items():
    summary_results.append({
        'Model_Pair': pair,
        'Test_R2_Mean': np.mean(metrics['Test_R2']),
        'Test_R2_STD': np.std(metrics['Test_R2']),
        'Test_MSE_Mean': np.mean(metrics['Test_MSE']),
        'Test_MSE_STD': np.std(metrics['Test_MSE']),
        'Test_MAE_Mean': np.mean(metrics['Test_MAE']),
        'Test_MAE_STD': np.std(metrics['Test_MAE']),
        'Test_R_Mean': np.mean(metrics['Test_R']),
        'Test_R_STD': np.std(metrics['Test_R']),
        'Test_Accuracy_Mean': np.mean(metrics['Test_Accuracy']),
        'Test_Accuracy_STD': np.std(metrics['Test_Accuracy']),
        'Test_MAPE_Mean': np.mean(metrics['Test_MAPE']),
        'Test_MAPE_STD': np.std(metrics['Test_MAPE']),
        'Train_R2_Mean': np.mean(metrics['Train_R2']),
        'Train_R2_STD': np.std(metrics['Train_R2']),
        'Train_MSE_Mean': np.mean(metrics['Train_MSE']),
        'Train_MSE_STD': np.std(metrics['Train_MSE']),
        'Train_MAE_Mean': np.mean(metrics['Train_MAE']),
        'Train_MAE_STD': np.std(metrics['Train_MAE']),
        'Train_R_Mean': np.mean(metrics['Train_R']),
        'Train_R_STD': np.std(metrics['Train_R']),
        'Train_Accuracy_Mean': np.mean(metrics['Train_Accuracy']),
        'Train_Accuracy_STD': np.std(metrics['Train_Accuracy']),
        'Train_MAPE_Mean': np.mean(metrics['Train_MAPE']),
        'Train_MAPE_STD': np.std(metrics['Train_MAPE'])
    })

summary_df = pd.DataFrame(summary_results)


summary_df = summary_df.sort_values(by='Test_R2_Mean', ascending=False).reset_index(drop=True)

print("\n所有模型组合的评估结果（平均值与标准差）：")
print(summary_df)
summary_df.to_csv("stacking_pair_repeated_holdout_summary1226.csv", index=False)


plt.figure(figsize=(14, 12))
sns.barplot(
    x='Test_R2_Mean', 
    y='Model_Pair', 
    data=summary_df, 
    palette='viridis',
    ci= 'sd'
)
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  
plt.xlabel('测试集 $R^2$ 平均分数')
plt.ylabel('模型组合')
plt.title('不同模型组合的测试集 $R^2$ 分数比较 (平均 ± 标准差)')
plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 12))
sns.barplot(
    x='Test_Accuracy_Mean', 
    y='Model_Pair', 
    data=summary_df, 
    palette='coolwarm',
    ci= 'sd'
)
plt.xlabel('测试集 Accuracy (±10%) 平均分数')
plt.ylabel('模型组合')
plt.title('不同模型组合的测试集 Accuracy (±10%) 分数比较 (平均 ± 标准差)')
plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 12))
sns.barplot(
    x='Test_MAPE_Mean', 
    y='Model_Pair', 
    data=summary_df, 
    palette='magma',
   ci= 'sd'
)
plt.xlabel('测试集 MAPE 平均分数 (%)')
plt.ylabel('模型组合')
plt.title('不同模型组合的测试集 MAPE 分数比较 (平均 ± 标准差)')
plt.tight_layout()
plt.show()


best_model_pair = summary_df.loc[summary_df['Test_R2_Mean'].idxmax(), 'Model_Pair']
print(f"最佳模型组合: {best_model_pair}")

import joblib

base_model_1 = best_models['model1']
base_model_2 = best_models['model2']


meta_model = RidgeCV()


stacking_regressor = StackingRegressor(
    estimators=[('model1', base_model_1), ('model2', base_model_2)],
    final_estimator=meta_model,
    cv=LeaveOneOut(),
    passthrough=False,
    n_jobs=-1
)


X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.n, random_state=24
)


stacking_regressor.fit(X_train, y_train)

y_pred_test = stacking_regressor.predict(X_test)
y_pred_train = stacking_regressor.predict(X_train)


mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)
r_test, _ = pearsonr(y_test, y_pred_test)  


mse_train = mean_squared_error(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)
r_train, _= pearsonr(y_train, y_pred_train)  


print("测试集 MSE:", mse_test)
print("测试集 MAE:", mae_test)
print("测试集 R²:", r2_test)
print("测试集 R:", r_test)

print("训练集 MSE:", mse_train)
print("训练集 MAE:", mae_train)
print("训练集 R²:", r2_train)
print("训练集 R:", r_train)


joblib.dump(stacking_regressor, 'best_stacking_model.pkl')


plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  
plt.plot(range(len(y_test)), y_test, color="blue", linewidth=1.5, linestyle="-")  
plt.plot(range(len(y_pred_test)), y_pred_test, color="red", linewidth=1.5, linestyle="-.")  
plt.legend(['真实值', '预测值'])  
plt.title("真实值与预测值比对图")  
plt.show()  


df_train = pd.DataFrame({
    "Dataset": "Train",
    "True_Value": y_train,
    "Predicted_Value": y_pred_train
})

df_test = pd.DataFrame({
    "Dataset": "Test",
    "True_Value": y_test,
    "Predicted_Value": y_pred_test
})

df_all = pd.concat([df_train, df_test], ignore_index=True)


df_all.to_excel("fit_result.xlsx", index=False)

 
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train, alpha=0.6)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title(f"Train Set: R² = {r2_score(y_train, y_pred_train):.2f}")

    
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title(f"Test Set: R² = {r2_score(y_test, y_pred_test):.2f}")

plt.tight_layout()
plt.show()

file_path1 = r'your file'
df1 = pd.read_excel(file_path1, dtype=str)


df_anchor = pd.read_excel(r'your file', dtype=str, sheet_name='Sheet1')
anchor_ids = df['anchor_ids'] = df_anchor['Anchoring'].apply(lambda x: list(map(int, x)))


df_linker = pd.read_excel(r'your file', dtype=str, sheet_name='Sheet2')
linker_ids = df['linker_ids'] = df_linker['Linker'].apply(lambda x: list(map(int, x)))

df_head = pd.read_excel(r'your file', dtype=str, sheet_name='Sheet3')
head_ids = df['head_ids'] = df_head['Head'].apply(lambda x: list(map(int, x)))


anchor_labels = [f'A{i+1}' for i in range(len(anchor_ids))]    
linker_labels = [f'L{i+1}' for i in range(len(linker_ids))]    
head_labels = [f'H{i+1}' for i in range(len(head_ids))]        

import itertools

combinations = list(itertools.product(range(len(anchor_ids)), range(len(linker_ids)), range(len(head_ids))))

full_code_list = []

for combo in combinations:
    a_idx, l_idx, h_idx = combo
    full_code = anchor_ids[a_idx] + linker_ids[l_idx] + head_ids[h_idx]
    full_code_list.append(full_code)


x_virtual = np.array(full_code_list)
scaler = StandardScaler()
x_virtual = scaler.fit_transform(x_virtual)

print(f"虚拟分子库的特征矩阵形状: {x_virtual.shape}")  

y_pred_virtual = stacking_regressor.predict(x_virtual)


df_virtual = pd.DataFrame(combinations, columns=['Anchor_Index', 'Linker_Index', 'Head_Index'])


df_virtual['Anchoring Group'] = df_virtual['Anchor_Index'].apply(lambda x: anchor_labels[x] if 'anchor_labels' in locals() else f'A{x+1}')
df_virtual['Linker Group'] = df_virtual['Linker_Index'].apply(lambda x: linker_labels[x] if 'linker_labels' in locals() else f'L{x+1}')
df_virtual['Functional Head'] = df_virtual['Head_Index'].apply(lambda x: head_labels[x] if 'head_labels' in locals() else f'H{x+1}')


df_virtual['Predicted PCE (%)'] = y_pred_virtual


df_virtual['Full Code'] = df_virtual.apply(lambda row: f"{row['Anchoring Group']}_{row['Linker Group']}_{row['Functional Head']}", axis=1)


df_virtual = df_virtual[['Anchoring Group', 'Linker Group', 'Functional Head', 'Full Code', 'Predicted PCE (%)']]


output_file = r'your file'
df_virtual.to_excel(output_file, index=False)

print(f"预测结果已保存到 {output_file}")

import seaborn as sns


file_path2 = r'your file'
df2 = pd.read_excel(file_path2)


print("数据预览:")
print(df2.head())


pce = df2['Predicted PCE (%)']
missing_values = pce.isnull().sum()
print(f"\n缺失值数量: {missing_values}")


pce = pce.dropna()


statistics = pce.describe()
print("\nPCE 预测值的统计描述:")
print(statistics)


sns.set(style="whitegrid")

plt.figure(figsize=(10, 6), dpi=1200)
sns.histplot(pce, bins=30, kde=True, color='skyblue', edgecolor='black')

plt.title('Predicted PCE (%) Distribution with KDE', fontsize=16)
plt.xlabel('Predicted PCE (%)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

plt.show()


output_image = r'./pce_distribution.png'
plt.savefig(output_image, dpi=1200, bbox_inches='tight')
print(f"PCE 分布图已保存到 {output_image}")







































































































































































from __future__ import annotations
import os
import math
import json
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")



DATA_PATH = r'C:\Users\omen\Desktop\one-hot.xlsx'   
SHEET_NAME = 0                       
TARGET_COL = "PCE/%"                   
ANCHOR_COL = "Anchoring Group"
LINKER_COL = "Linker Group"
FUNC_COL   = "Functional Head"


MODEL_TYPE = "rf"
RANDOM_STATE = 42
TEST_SIZE = 0.2



FRAGMENT_NAME_MAP_ANCHORING: Dict[int, str] = {}
FRAGMENT_NAME_MAP_LINKER: Dict[int, str] = {}
FRAGMENT_NAME_MAP_FUNCTIONAL: Dict[int, str] = {}


OUT_DIR = "./shap_outputs"
os.makedirs(OUT_DIR, exist_ok=True)




def _read_table(data_path: str, sheet=0) -> pd.DataFrame:
    ext = os.path.splitext(data_path)[1].lower()
    if ext in [".xls", ".xlsx"]:
        
        df = pd.read_excel(data_path, sheet_name=sheet, dtype={ANCHOR_COL: str, LINKER_COL: str, FUNC_COL: str})
    elif ext == ".csv":
        df = pd.read_csv(data_path, dtype={ANCHOR_COL: str, LINKER_COL: str, FUNC_COL: str})
    else:
        raise ValueError("Only .xlsx/.xls/.csv are supported")
    
    for c in [ANCHOR_COL, LINKER_COL, FUNC_COL]:
        df[c] = df[c].astype(str).str.strip().replace({"nan": "", "None": ""})
    return df


def _infer_width(series: pd.Series) -> int:
    
    return int(series.map(lambda s: len(str(s))).max())


def _split_digits(s: str, width: int) -> List[int]:
    s = (s or "").strip()
    s = s if s else ""
    
    s = s.zfill(width)
    return [int(ch) for ch in s]


def _expand_block(df: pd.DataFrame, col: str, prefix: str, name_map: Dict[int, str]) -> pd.DataFrame:
    width = _infer_width(df[col])
    values = df[col].apply(lambda s: _split_digits(s, width))
    block = pd.DataFrame(values.tolist(), index=df.index)
    
    cols = []
    for i in range(width):
        human = name_map.get(i+1)
        cols.append(f"{prefix}_{human if human else i+1}")
    block.columns = cols
    return block


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    anch = _expand_block(df, ANCHOR_COL, "Anchoring", FRAGMENT_NAME_MAP_ANCHORING)
    link = _expand_block(df, LINKER_COL, "Linker", FRAGMENT_NAME_MAP_LINKER)
    func = _expand_block(df, FUNC_COL,   "Functional", FRAGMENT_NAME_MAP_FUNCTIONAL)
    X = pd.concat([anch, link, func], axis=1)
    return X, list(anch.columns), list(link.columns), list(func.columns)



def make_model(kind: str):
    kind = kind.lower()
    if kind == "rf":
        return RandomForestRegressor(n_estimators=500, max_depth=None, n_jobs=-1, random_state=RANDOM_STATE)
    elif kind == "xgb":
        import xgboost as xgb
        return xgb.XGBRegressor(
            n_estimators=1000, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist"
        )
    elif kind == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=1000, learning_rate=0.05, max_depth=-1,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=-1
        )
    elif kind == "linear":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0, random_state=RANDOM_STATE)
    else:
        raise ValueError("MODEL_TYPE must be one of: rf/xgb/lgbm/linear")



def train_and_eval(X: pd.DataFrame, y: pd.Series):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    model = make_model(MODEL_TYPE)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    metrics = {
        "R2": r2_score(yte, pred),
        "MAE": float(mean_absolute_error(yte, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(yte, pred)))
    }
    return model, (Xtr, Xte, ytr, yte), metrics



def compute_shap(model, X: pd.DataFrame):
    import shap
    
    try:
        import xgboost as _xgb  
        import lightgbm as _lgb 
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        expected_value = explainer.expected_value
    except Exception:
        try:
            explainer = shap.LinearExplainer(model, X)
            shap_values = explainer.shap_values(X)
            expected_value = explainer.expected_value
        except Exception:
            
            explainer = shap.KernelExplainer(model.predict, shap.sample(X, 200))
            shap_values = explainer.shap_values(shap.sample(X, 500))
            expected_value = explainer.expected_value
    return explainer, shap_values, expected_value


def export_shap_tables(X: pd.DataFrame, shap_values: np.ndarray):
    
    mean_abs = np.abs(shap_values).mean(axis=0)
    feat_df = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    feat_df.to_csv(os.path.join(OUT_DIR, "feature_level_shap.csv"), index=False)

    
    blocks = {"Anchoring": [], "Linker": [], "Functional": []}
    for c, v in zip(X.columns, mean_abs):
        if c.startswith("Anchoring_"): blocks["Anchoring"].append(v)
        elif c.startswith("Linker_"):   blocks["Linker"].append(v)
        elif c.startswith("Functional_"): blocks["Functional"].append(v)
    block_df = pd.DataFrame({
        "block": list(blocks.keys()),
        "mean_abs_shap_sum": [float(np.sum(v)) for v in blocks.values()]
    }).sort_values("mean_abs_shap_sum", ascending=False)
    block_df.to_csv(os.path.join(OUT_DIR, "block_level_shap.csv"), index=False)

    return feat_df, block_df


def plot_shap_summary(X: pd.DataFrame, shap_values: np.ndarray):
    import shap
    import matplotlib.pyplot as plt
    
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_summary_dot.png"), dpi=300, bbox_inches="tight")
    plt.close()

    
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_summary_bar.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_block_bar(block_df: pd.DataFrame):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6,4))
    plt.bar(block_df["block"], block_df["mean_abs_shap_sum"])
    plt.ylabel("Sum of mean |SHAP|")
    plt.title("Block-level importance (Anchoring vs Linker vs Functional)")
    for idx, val in enumerate(block_df["mean_abs_shap_sum"]):
        plt.text(idx, val, f"{val:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "block_level_importance.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_force_example(explainer, shap_values: np.ndarray, X: pd.DataFrame, idx: int = 0):
    import shap
    
    out_html = os.path.join(OUT_DIR, f"force_plot_idx{idx}.html")
    shap.save_html(out_html, shap.force_plot(explainer.expected_value, shap_values[idx,:], X.iloc[idx,:]))
    return out_html



if __name__ == "__main__":
    print("[1/6] 读取数据...")
    df = _read_table(DATA_PATH, SHEET_NAME)
    assert TARGET_COL in df.columns, f"目标列 {TARGET_COL} 不存在。当前列：{list(df.columns)}"

    print("[2/6] 构建特征矩阵...")
    X, anch_cols, link_cols, func_cols = build_features(df)
    y = df[TARGET_COL].astype(float)
    print(f"X shape = {X.shape}; y = {y.shape}")

    print("[3/6] 训练模型...")
    model, splits, metrics = train_and_eval(X, y)
    print("Metrics:", json.dumps(metrics, ensure_ascii=False, indent=2))
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("[4/6] 计算 SHAP 值...")
    explainer, shap_values, expected_value = compute_shap(model, X)

    print("[5/6] 导出表格与图像...")
    feat_df, block_df = export_shap_tables(X, shap_values)
    plot_shap_summary(X, shap_values)
    plot_block_bar(block_df)
    fp = plot_force_example(explainer, shap_values, X, idx=0)

    print("[6/6] 完成！输出文件：")
    for fn in [
        "metrics.json",
        "feature_level_shap.csv",
        "block_level_shap.csv",
        "shap_summary_dot.png",
        "shap_summary_bar.png",
        "block_level_importance.png",
        os.path.basename(fp)
    ]:
        print(" -", os.path.join(OUT_DIR, fn))

    
    if any([FRAGMENT_NAME_MAP_ANCHORING, FRAGMENT_NAME_MAP_LINKER, FRAGMENT_NAME_MAP_FUNCTIONAL]):
        print("\n已检测到片段名称映射，以下示例展示如何聚合到人类可读片段：")
        
        mean_abs = np.abs(shap_values).mean(axis=0)
        s = pd.Series(mean_abs, index=X.columns)
        
        agg = {}
        for col, val in s.items():
            if col.startswith("Anchoring_"):
                frag = col[len("Anchoring_"):]
                agg[("Anchoring", frag)] = agg.get(("Anchoring", frag), 0.0) + float(val)
            elif col.startswith("Linker_"):
                frag = col[len("Linker_"):]
                agg[("Linker", frag)] = agg.get(("Linker", frag), 0.0) + float(val)
            elif col.startswith("Functional_"):
                frag = col[len("Functional_"):]
                agg[("Functional", frag)] = agg.get(("Functional", frag), 0.0) + float(val)
        agg_df = pd.DataFrame(
            [(k[0], k[1], v) for k, v in agg.items()], columns=["block", "fragment", "sum_mean_abs_shap"]
        ).sort_values("sum_mean_abs_shap", ascending=False)
        agg_df.to_csv(os.path.join(OUT_DIR, "fragment_name_aggregated_shap.csv"), index=False)
        print(" -", os.path.join(OUT_DIR, "fragment_name_aggregated_shap.csv"))


