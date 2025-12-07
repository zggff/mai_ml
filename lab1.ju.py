# %% [md]
# ## Гига М.Я. М8О-313Б-23

# %% [md]
# MSE = 24.99

# %%
from typing import List, Tuple, cast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler, OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import os
import seaborn as sns
from scipy import stats
import scipy
from scipy.stats import norm, skew
from scipy.special import inv_boxcox
from sklearn.mixture import GaussianMixture
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

pd.options.display.max_columns = None

import warnings
warnings.filterwarnings("ignore")

# %%
def optimal_weight(y, pred_a, pred_b):
    d = pred_a - pred_b
    denom = np.dot(d, d)
    if denom == 0:
        return 0.5
    w = float(np.dot(d, (y - pred_b)) / denom)
    return float(np.clip(w, 0.0, 1.0))


# %% [md]
# Преобразуем dataframe, убираем nan, убираем некорректные значения
# %%
df: pd.DataFrame = pd.read_csv("./data/train.csv")
df = df[~df["RiskScore"].isna()]
df = cast(pd.DataFrame, df[df["RiskScore"].abs() < 200])
df.head()  # pyright: ignore[reportUnusedExpression]


# %%
for col in df.select_dtypes(include=[np.number]).columns:
    sns.histplot(df[col])
    plt.title(col)
    plt.show()

# %%
numerical_data = df[df.select_dtypes(include=np.number).columns]
plt.figure(figsize=(10, 8))
sns.heatmap(numerical_data.corr(), annot=False, cmap="viridis")

# %% [md]
# зависимости есть. Надо будет убрать

# %% [md]
# проверяем min max и z-score

# %%
dfNum = df.select_dtypes(include=np.number)

def z_score_scaler(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / (std)


def min_max_scaler(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    return (data - min_val) / (max_val - min_val)

dfNum.apply(z_score_scaler).describe()
# %%
dfNum.apply(min_max_scaler).describe()

# %%
def preprocess(X_in: pd.DataFrame) -> pd.DataFrame:
    X = X_in.copy()
    categorical = X.select_dtypes(include=["object"]).columns
    numerical = X.select_dtypes(include=[np.number]).columns

    for col in numerical:
        if X[col].isnull().sum() == 0:
            continue
        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)

    for col in categorical:
        if X[col].isnull().sum() == 0:
            continue
        mode_val = X[col].mode()[0]
        X[col].fillna(mode_val, inplace=True)


    X["LoanToIncomeRatio"] = X["LoanAmount"] / (X["AnnualIncome"] + 1)
    X["DebtToAssetsRatio"] = X["TotalLiabilities"] / (X["TotalAssets"] + 1)
    X["SavingsToLoanRatio"] = X["SavingsAccountBalance"] / (X["LoanAmount"] + 1)
    X["CreditHistoryInteraction"] = X["LengthOfCreditHistory"] * X["PaymentHistory"]

    clip_cols = ["MonthlyIncome","LoanAmount","AnnualIncome","SavingsAccountBalance","TotalAssets","TotalLiabilities","MonthlyDebtPayments"]
    for col in clip_cols:
        lo = X[col].quantile(0.01)
        hi = X[col].quantile(0.99)
        X[col] = X[col].clip(lo, hi)
    log_candidates = ["MonthlyIncome","LoanAmount","SavingsAccountBalance","CheckingAccountBalance","TotalAssets","TotalLiabilities","NetWorth","MonthlyDebtPayments"]
    for col in log_candidates:
        X[col] = np.log1p(X[col])

    return X


y = df["RiskScore"]
X = df.drop(columns=["RiskScore"])

X_eng = preprocess(X)
X_eng.head()

# %% [md]
# убираем зависимости
# %%
numericalX = X_eng.select_dtypes(include=[np.number]).copy()
corr = numericalX.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
X_eng.drop(columns=to_drop)
to_drop



# %%
binned = ["Age", "CreditScore"]
categorical = X.select_dtypes(include=["object"]).columns
numerical = [x for x in X.select_dtypes(include=[np.number]).columns if x not in binned]

numerical_transformer = Pipeline(steps=[
    ("scaler", RobustScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True, drop="first"))
])

binned_transformer = Pipeline(steps=[
    ("bin", KBinsDiscretizer(n_bins=20, encode="onehot-dense", strategy="quantile"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical),
        ("cat", categorical_transformer, categorical),
        ("bin", binned_transformer, binned)
    ])


pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("var", VarianceThreshold(threshold=1e-5)),
    ("preselect", SelectKBest(score_func=f_regression, k=X_eng.shape[1])),
    ("poly", PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)),
    ("select", SelectKBest(score_func=f_regression, k=800)),
])
pipe

# %%
y_bins = pd.cut(y, bins=6, labels=False)
X_train, X_test, y_train, y_test = train_test_split(X_eng, y, test_size=0.3, random_state=42, stratify=y_bins)

X_train = pipe.fit_transform(X_train, y_train)
X_test   = pipe.transform(X_test)


# %% [md]
# на предыдущей 

# alpha=28.31578947368421, w=0.6166562093103263, mse=27.044682632237002

# alpha=27.846153846153847, w=0.6147406385774055, mse=27.044760042886168
# alpha=27.692307692307693, w=0.6140767374162226, mse=27.0447425348076

# %%

# %%

# alpha_grid = np.linspace(0, 100, 20)
# alpha_grid = np.linspace(10, 100, 100)
# alpha_grid = np.linspace(20, 29, 40)
# alpha_grid = np.linspace(27, 28, 40)
alpha_grid = [27.692307692307693]

# alpha_grid = [28.205128205128204]
alpha_grid

best_mse = 2000
best_w = 1
best_alpha = 0

for i, alpha in enumerate(alpha_grid):
    rid = Ridge(alpha=float(alpha))
    rid.fit(X_train, y_train)

    lin = LinearRegression()
    lin.fit(X_train, y_train)

    lin_pred = lin.predict(X_test)
    rig_pred = rid.predict(X_test)

    weight = optimal_weight(y_test.values, lin_pred, rig_pred)
    diff = 0.05
    step = 0.001
    weights = np.clip(np.arange(weight - diff, weight + diff + step, step), 
                      0.0, 1.0)

    diff = None
    for w in weights:
        pred_mix = w * lin_pred + (1.0 - w) * rig_pred
        mse = mean_squared_error(y_test, pred_mix)
        if mse < best_mse:
            diff = best_mse - mse
            best_alpha = alpha
            best_mse = mse
            best_w = w

    diff = 0 if diff is None else diff
    print(f"alpha  = {alpha:.4f} [{i+1:2}/{len(alpha_grid)}]: {diff}")

print(f"alpha={best_alpha}, w={best_w}, mse={best_mse}")

# %% [md]
# решаем задачу
# %%
X_full = X_eng.copy()
X_out = pd.read_csv("./data/test.csv")
X_out = X_out.drop(columns="ID")
X_out = preprocess(X_out)

pipe_final = pipe
X_full = pipe_final.fit_transform(X_full, y)
X_out = pipe_final.transform(X_out)

lin = LinearRegression()
lin.fit(X_full, y)

rig = Ridge(alpha=best_alpha)
rig.fit(X_full, y)

lin_pred = lin.predict(X_out)
rig_pred = rig.predict(X_out)
mix_pred = best_w * lin_pred + (1.0 - best_w) * rig_pred

sub = pd.DataFrame({"ID": np.arange(len(mix_pred)), "RiskScore": mix_pred})
sub.to_csv("out.csv", index=False)
sub.sort_values(by="RiskScore")


# %%
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mape(y_true, y_pred, epsilon=1e-10):
    return np.mean(np.abs((y_true - y_pred) / y_true + epsilon)) 

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return 1 - (ss_res / ss_tot)


lin = LinearRegression()
lin.fit(X_train, y_train)
pred = lin.predict(X_test)
print(abs(mean_squared_error(y_test, pred) - mse(y_test, pred)))
print(abs(mean_absolute_error(y_test, pred) - mae(y_test, pred)))
print(abs(mean_absolute_percentage_error(y_test, pred) - mape(y_test, pred)))
print(abs(r2_score(y_test, pred) - r2(y_test, pred)))


# %% [md]
# класс линейной регрессии

# %%
class Linear:
    def _fit_analyze(self, X, y):
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y

    def _fit_gradient(self, X, y, learning_rate, n_iterations):
        h, w = X.shape
        self.weights = np.random.randn(w) * 0.01
        for i in range(n_iterations):
            pred = X @ self.weights
            gradient = (2 / h) * X.T @ (pred - y)
            self.weights -= learning_rate * gradient
            if np.any(np.isnan(self.weights)) or np.any(np.isinf(self.weights)):
                raise RuntimeError(f"Warning: NaN/Inf detected at iteration {i}\n" +
                                    f"Gradient norm: {np.linalg.norm(gradient)}")

    def _fit_stoch(self, X, y, learning_rate, n_iterations):
        h, w = X.shape
        self.weights = np.random.randn(w) * 0.01
        for _ in range(n_iterations):
            indices = np.random.permutation(h)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            for i in range(h):
                xi = X[i:i+1]
                yi = y[i:i+1]
                pred = xi @ self.weights
                gradient = 2 * xi.T @ (pred - yi)
                self.weights -= learning_rate * gradient
                if np.any(np.isnan(self.weights)) or np.any(np.isinf(self.weights)):
                    raise RuntimeError(f"Warning: NaN/Inf detected at iteration {i}\n" +
                                        f"Gradient norm: {np.linalg.norm(gradient)}")


    def fit(self, X, y, method = 0, learning_rate = 0.01, n_iterations = 1000):
        if method == 0:
            return self._fit_analyze(X, y)
        if method == 1:
            return self._fit_gradient(X, y, learning_rate, n_iterations)
        if method == 2:
            return self._fit_stoch(X, y, learning_rate, n_iterations)
        raise RuntimeError(f"invalid method")

    def predict(self, X):
        if self.weights is None:
            raise RuntimeError(f"not taught")
        return X @ self.weights

# %% [md]
# проверяем работу класса

# %%
df2 = df.copy().select_dtypes(include=np.number).dropna()
df2 = df2.apply(z_score_scaler)
y2 = df2["RiskScore"]
X2 = df2.drop(columns=["RiskScore"])

X_train, X_test, y_train, y_test = train_test_split(X2.to_numpy(), y2.to_numpy(), test_size=0.33, random_state=42)


# %%
# %%time
lin_sk = LinearRegression()
lin_sk.fit(X_train, y_train)
pred = lin_sk.predict(X_test)
mse(y_test, pred)

# %%
# %%time
lin_an = Linear()
lin_an.fit(X_train, y_train, 0)
pred = lin_an.predict(X_test)
mse(y_test, pred)

# %%
# %%time
lin_gr = Linear()
lin_gr.fit(X_train, y_train, 1, n_iterations=10000)
pred = lin_gr.predict(X_test)
mse(y_test, pred)

# %%
lin_sgr = Linear()
lin_sgr.fit(X_train, y_train, 2, n_iterations=1000)
pred = lin_sgr.predict(X_test)
mse(y_test, pred)

# %%
res = pd.DataFrame()
res["fields"] = X2.columns
res["sklearn"] = lin_sk.coef_
res["analytical"] = lin_an.weights
res["gradient"] = lin_gr.weights
res["stoch"] = lin_sgr.weights
res

# %% [md]
# аналитическая и sklearn совпадают. Спуски немного отличаются

