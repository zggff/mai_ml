# %% [md]
# ## Гига М.Я. М8О-313Б-23

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelEncoder
# from sklearn.linear_model import LinearRegression


# %% [md]
# тестирование результата
# %%
def mse():
    pass


# %%
df = pd.read_csv("./data/train.csv")
df = df.dropna()
df = df[df['RiskScore'].abs() < 9999999]
df = df.sort_values(by="RiskScore")
df["ApplicationDate"] = pd.to_datetime(df['ApplicationDate'])


df

# %%

plt.figure(figsize=[10, (df.shape[1] - 1) * 5])
for i, column in enumerate(df.columns):
    if column == "RiskScore":
        continue
    plt.subplot((df.shape[1]) // 2 , 2, i + 1)
    plt.scatter(df[column], df["RiskScore"])
    plt.title(column)

# %% [md]
# По графикам видно, что можно исключить JobTenure, EmploymentStatus,
# EducationLevel, Experience, LengthOfCreditHistory, 
# PreviousLoanDefaults, PaymentHistory, BankruptcyHistory,
# LoanPurpose, HomeOwnershipStatus, MaritalStatus, LoadDuration, 
# ApplicationDate

