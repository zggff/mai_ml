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


# %% [md]
# Преобразуем dataframe, убираем nan, убираем некорректные значения
# %%
df: pd.DataFrame = pd.read_csv("./data/train.csv")
df = df.dropna()
df = cast(pd.DataFrame, df[df['RiskScore'].abs() < 200])
df["ApplicationDate"] = pd.to_datetime(df['ApplicationDate'])
df.sort_values(by="RiskScore") # pyright: ignore[reportUnusedExpression]


# %%
plt.figure(figsize=[10, df.shape[1] * 5])
for i, column in enumerate(df.columns):
    if column == "RiskScore":
        continue
    plt.subplot(round(df.shape[1] / 2), 2, i + 1)
    plt.scatter(df[column], df["RiskScore"])
    plt.title(column)


# %% [md]
# По графикам видно, что можно исключить JobTenure, EmploymentStatus,
# EducationLevel, Experience, LengthOfCreditHistory,
# PreviousLoanDefaults, PaymentHistory, BankruptcyHistory,
# LoanPurpose, HomeOwnershipStatus, MaritalStatus, LoadDuration,
# ApplicationDate


# %%

categoric = ["EducationLevel", "EmploymentStatus", 
             "HomeOwnershipStatus",
             "MaritalStatus", "LoanPurpose"]

df2 = df.copy()
le = LabelEncoder()
for col in categoric:
    df2[col] = le.fit_transform(df2[col])
correlation_matrix = df2.corr()
print(correlation_matrix.sort_values(by="RiskScore")["RiskScore"])



# %%
def KTest(X, y) -> Tuple[LinearRegression, List[float]]:
    scores = []
    model_best = LinearRegression(fit_intercept=False)
    max_score = 1000

    cv = KFold(n_splits=10, shuffle=True)
    for train_index, test_index in cv.split(X_ohe):
        model = LinearRegression(fit_intercept=False)
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append(mean_squared_error(y_test, y_pred))
        if scores[-1] < max_score:
            model_best = model
            max_score = scores[-1]
    return model_best, scores


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    selected = ["CreditScore", "MonthlyIncome", "BaseInterestRate", "TotalDebtToIncomeRatio"]
    res = cast(pd.DataFrame, df[selected + categoric])
    # repeats = ["AnnualIncome", "BaseInterestRate"]
    # res = df.drop(columns=repeats)
    return res


y = df['RiskScore']
X = df.drop(columns=['RiskScore'])
X = preprocess(X)
enc = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
X_ohe = enc.fit_transform(X)


model, scores = KTest(X_ohe, y)
scores

# model.fit(X_train, y_train)
# print("fitted model")


# %%
# X_train, X_test, y_train, y_test = train_test_split(X_ohe, y, test_size=0.1, random_state=1)
#
y_pred = model.predict(X_ohe)
print(f"MSE  = {mean_squared_error(y, y_pred)}")
print(f"MAE  = {mean_absolute_error(y, y_pred)}")
print(f"MAPE = {mean_absolute_percentage_error(y, y_pred)}")
print(f"R^2  = {r2_score(y, y_pred)}")

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
X_valid = enc.transform(X_valid)
print("prepared")
y_valid = model.predict(X_valid)
res = pd.DataFrame()
res["ID"] = np.arange(len(y_valid))
res["RiskScore"] = y_valid 
res.set_index("ID", inplace=True)

os.remove("./out.csv")
res.to_csv("./out.csv")
print("written")
