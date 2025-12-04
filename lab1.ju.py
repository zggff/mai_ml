# %% [md]
# ## Гига М.Я. М8О-313Б-23

# %%
from typing import List, Tuple, cast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.feature_selection import RFE
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


# %% [md]
# Преобразуем dataframe, убираем nan, убираем некорректные значения
# %%
df: pd.DataFrame = pd.read_csv("./data/train.csv")
df = df.dropna()
df = cast(pd.DataFrame, df[df['RiskScore'].abs() < 200])
df["ApplicationDate"] = pd.to_datetime(df['ApplicationDate'])
df.sort_values(by="RiskScore")  # pyright: ignore[reportUnusedExpression]


# %%
plt.figure(figsize=[10, df.shape[1] * 5])
for i, column in enumerate(df.columns):
    if column == "RiskScore":
        continue
    plt.subplot(round(df.shape[1] / 2), 2, i + 1)
    plt.scatter(df[column], df["RiskScore"])
    plt.title(column)



# %%
y = df['RiskScore']
X = df.drop(columns=['RiskScore'])
categorical = X.select_dtypes(include=['object']).columns
numerical = X.select_dtypes(include=[np.number]).columns
le_dict = {}
print("created X")


# %%
def preprocess(X: pd.DataFrame) -> pd.DataFrame:
    X_eng = X.copy()

    # Create interaction features
    X_eng['IncomeToLoanRatio'] = X_eng['AnnualIncome'] / (X_eng['LoanAmount'] + 1)
    X_eng['MonthlyIncomeToDebt'] = X_eng['MonthlyIncome'] / (X_eng['MonthlyDebtPayments'] + 1)
    X_eng['CreditUtilizationToScore'] = X_eng['CreditCardUtilizationRate'] * X_eng['CreditScore']
    X_eng['AssetToLiabilities'] = X_eng['TotalAssets'] / (X_eng['TotalLiabilities'] + 1)
    X_eng['DebtBurden'] = X_eng['TotalDebtToIncomeRatio'] * X_eng['DebtToIncomeRatio']
    X_eng['AgeIncomeInteraction'] = X_eng['Age'] * X_eng['AnnualIncome'] / 1000

    key_numerical = ['CreditScore', 'AnnualIncome', 'LoanAmount', 'MonthlyIncome', 
                     'DebtToIncomeRatio', 'CreditCardUtilizationRate']

    for col in key_numerical:
        if col in X_eng.columns:
            X_eng[f'{col}_squared'] = X_eng[col] ** 2
            X_eng[f'{col}_log'] = np.log1p(np.abs(X_eng[col]))

    numerical_cols_eng = X_eng.select_dtypes(include=[np.number]).columns.tolist()

    for col in numerical_cols_eng:
        if X_eng[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            q99 = X_eng[col].quantile(0.99)
            if q99 > X_eng[col].median() * 10:  # Only cap if there are extreme outliers
                X_eng[col] = np.where(X_eng[col] > q99, q99, X_eng[col])
    return X_eng


X_eng = preprocess(X)
numerical_cols_eng = X_eng.select_dtypes(include=[np.number]).columns.tolist()
print("preprocessed")

# %%
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols_eng),
        ('cat', categorical_transformer, categorical)
    ])


# %%

X_train, X_test, y_train, y_test = train_test_split(
    X_eng, y, test_size=0.2, random_state=42
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])
pipeline.fit(X_train, y_train)


# %%
y_pred = pipeline.predict(X_test)
print(f"MSE  = {mean_squared_error(y_test, y_pred)}")
print(f"MAE  = {mean_absolute_error(y_test, y_pred)}")
print(f"MAPE = {mean_absolute_percentage_error(y_test, y_pred)}")
print(f"R^2  = {r2_score(y_test, y_pred)}")

# %%
plt.hist(y_test, bins=30, alpha=0.5, label='Actual', density=True)
plt.hist(y_pred, bins=30, alpha=0.5, label='Predicted', density=True)
plt.legend()

# %%
path = "./out.csv"
X_valid = pd.read_csv("./data/test.csv")
X_valid = X_valid.drop(columns=["ID"])
X_valid = preprocess(X_valid)

y_valid = pipeline.predict(X_valid)
res = pd.DataFrame()
res["ID"] = np.arange(len(y_valid))
res["RiskScore"] = y_valid
res.set_index("ID", inplace=True)

os.remove("./out.csv")
res.to_csv("./out.csv")
print("written")
