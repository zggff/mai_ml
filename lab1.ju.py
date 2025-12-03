# %% [md]
# ## Гига М.Я. М8О-313Б-23

# %%
from typing import List, Tuple, cast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import os
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew


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


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    res = df.drop(columns=["ApplicationDate"])
    for col in categorical:
        le = LabelEncoder()
        res[col] = le.fit_transform(res[col])
        le_dict[col] = le
    return res


X = preprocess(X)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)


# %%
y_pred = model.predict(X_test)
print(f"MSE  = {mean_squared_error(y_test, y_pred)}")
print(f"MAE  = {mean_absolute_error(y_test, y_pred)}")
print(f"MAPE = {mean_absolute_percentage_error(y_test, y_pred)}")
print(f"R^2  = {r2_score(y_test, y_pred)}")

# %%
plt.subplot()
n = 1000
plt.scatter(np.arange(n), y[:n], label="true")
plt.scatter(np.arange(n), y_pred[:n], label="prediction")
plt.legend()

# %%
path = "./out.csv"
X_valid = pd.read_csv("./data/test.csv")
X_valid = X_valid.drop(columns=["ID"])
X_valid = preprocess(X_valid)
print("prepared")
y_valid = model.predict(X_valid)
res = pd.DataFrame()
res["ID"] = np.arange(len(y_valid))
res["RiskScore"] = y_valid
res.set_index("ID", inplace=True)

os.remove("./out.csv")
res.to_csv("./out.csv")
print("written")
