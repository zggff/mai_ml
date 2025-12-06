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
from sklearn.preprocessing import PolynomialFeatures

pd.options.display.max_columns = None
epsilon = 1e-6


# %% [md]
# Преобразуем dataframe, убираем nan, убираем некорректные значения
# %%
df: pd.DataFrame = pd.read_csv("./data/train.csv")
df = df[~df['RiskScore'].isna()]
df = cast(pd.DataFrame, df[df['RiskScore'].abs() < 200])
df.head()  # pyright: ignore[reportUnusedExpression]


# %%
# numerical_data = df[df.select_dtypes(include=np.number).columns]
# plt.figure(figsize=(10, 8))
# sns.heatmap(numerical_data.corr(), annot=False, cmap='viridis')



# %%
def preprocess(X_in: pd.DataFrame) -> pd.DataFrame:
    X = X_in.copy()
    
    X = X.drop(columns=["ApplicationDate", "AnnualIncome"])
    categorical = X.select_dtypes(include=['object']).columns
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



    X['Income_to_Loan'] = X['MonthlyIncome'] / (X['LoanAmount'] + epsilon)
    X['Income_Debt_Ratio'] = X['MonthlyIncome'] / (X['MonthlyDebtPayments'] + epsilon)
    X['Assets_Liabilities_Ratio'] = X['TotalAssets'] / (X['TotalLiabilities'] + epsilon)
    X['NetWorth_Income'] = X['NetWorth'] / (X['MonthlyIncome'] + epsilon)
    X['Savings_Checking'] = X['SavingsAccountBalance'] + X['CheckingAccountBalance']
    X['Payment_Income_Ratio'] = X['MonthlyLoanPayment'] / (X['MonthlyIncome'] + epsilon)
    X['MonthlyIncome_Loan_Ratio'] = X['MonthlyIncome'] / (X['LoanAmount'] + epsilon)
    X['Assets_Income_Ratio'] = X['TotalAssets'] / (X['MonthlyIncome'] + epsilon)
    X['NetWorth_Loan_Ratio'] = X['NetWorth'] / (X['LoanAmount'] + epsilon)
    X['NetWorth_Loan_Ratio'] = X['NetWorth'] / (X['LoanAmount'] + epsilon)
    X['Debt_Loan_Ratio'] = X['MonthlyDebtPayments'] / (X['LoanAmount'] + epsilon)
    X['Savings_Loan_Ratio'] = X['SavingsAccountBalance'] / (X['LoanAmount'] + epsilon)
    X['Checking_Income_Ratio'] = X['CheckingAccountBalance'] / (X['MonthlyIncome'] + epsilon)
    X['Liabilities_Income_Ratio'] = X['TotalLiabilities'] / (X['MonthlyIncome'] + epsilon)
    X['Income_sqrt'] = np.sqrt(X['MonthlyIncome'])
    X['Loan_sqrt'] = np.sqrt(X['LoanAmount'])
    X['Assets_sqrt'] = np.sqrt(X['TotalAssets'])

    tosquare = ['Age',  'CreditScore', 'LoanAmount',
                     'DebtToIncomeRatio', 'CreditCardUtilizationRate',
                     'MonthlyIncome', 'MonthlyDebtPayments']

    tolog = ['CreditScore', 'DebtToIncomeRatio', 'Age', 'CreditCardUtilizationRate']

    tocube = ['CreditScore', 'DebtToIncomeRatio', 'Age']
    for col in tocube:
        X[f'{col}_cubed'] = X[col] ** 3

    for col in tolog:
        X[f'{col}_logged'] = np.log1p(X[col])

    for col in tosquare:
        X[f'{col}_squared'] = X[col] ** 2

    numerical_new = X.select_dtypes(include=[np.number]).columns.tolist()
    numerical_new = [col for col in numerical_new if col not in numerical]

    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)

    poly_features = poly.fit_transform(X[numerical])
    poly_vals = pd.DataFrame(
        poly_features,
        columns=poly.get_feature_names_out(numerical),
        index=X.index
    )

    X = pd.concat([X[categorical], X[numerical_new], poly_vals], axis=1)

    return X


y = df['RiskScore']
X = df.drop(columns=['RiskScore'])

X_eng = preprocess(X)
categorical = X_eng.select_dtypes(include=['object']).columns
numerical = X_eng.select_dtypes(include=[np.number]).columns
for col in X_eng.columns:
    cnt = X_eng[col].isna().sum()
    if cnt > 0:
        print(col, cnt)
X_eng.head()


# %%
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical),
        ('cat', categorical_transformer, categorical)
    ])


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])


# %%
mse = []

kf = KFold(n_splits=8, shuffle=True, random_state=42)
for train_index, test_index in kf.split(X_eng, y):
    pipeline.fit(X_eng.iloc[train_index], y.iloc[train_index])
    y_pred = pipeline.predict(X_eng.iloc[test_index])
    y_test = y.iloc[test_index]
    mse.append(mean_squared_error(y_test, y_pred))

    print(f"MSE  = {mean_squared_error(y_test, y_pred)}")

mse = np.average(mse)


# %%
X_train, X_test, y_train, y_test = train_test_split(X_eng, y, test_size=0.33, random_state=42)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
plt.text(0.1, 0.1, f"MSE  = {mse}", fontsize=10, transform=plt.gca().transAxes)
plt.hist(y_test, bins=30, alpha=0.5, label='Actual', density=True)
plt.hist(y_pred, bins=30, alpha=0.5, label='Predicted', density=True)
plt.legend()
plt.show()

# %%
pipeline.fit(X_eng, y)
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
