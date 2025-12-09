# %% [md]
# ## Гига М.Я. М80-313Б-33

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from typing import Any, Tuple, cast, Literal
from numpy import typing as npt

pd.options.display.max_columns = None

# import warnings
# warnings.filterwarnings("ignore")

# %% [md]
# utility functions


# %%
def z_score_scaler(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / (std)


def cont_to_binary(data: np.ndarray) -> np.ndarray:
    return (data >= 0.5).astype(int)


# %%

df_with_na = pd.read_csv("./data/train_c.csv")
df_with_na.describe()

# %%
null_info = df_with_na.isna().apply(lambda x: x.sum())
null_info  # pyright: ignore[reportUnusedExpression]

# %%
df_filled = df_with_na.dropna()
X_raw = df_filled.drop(columns=["LoanApproved"])
y_raw = cast(pd.Series, df_filled["LoanApproved"])

# %%
sns.heatmap(df_filled.select_dtypes(include=np.number).corr(),
            annot=False,
            cmap="viridis")

# %%
if False:
    for col in df_filled.select_dtypes(include=np.number).columns:
        sns.histplot(x=df_filled[col])
        plt.title(col)
        plt.show()


# %%
class MyBaggingRegressor:

    def __init__(self,
                 estimator: Any = LinearRegression,
                 n_estimators: int = 10,
                 sample_size: int = 1,
                 random_state: int | None = None):
        self.est = estimator
        self.n_estimators = n_estimators
        self.sample_size = sample_size
        if random_state is not None:
            np.random.seed(random_state)

    def _prepare(self, X: np.ndarray,
                 y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = X.shape[0]
        size = n_samples * self.sample_size
        indices = np.random.choice(n_samples, size=size, replace=True)
        return X[indices], y[indices]

    def _dupmodel(self):
        return self.est.__class__(**self.est.get_params())

    def _conv(self, val: Any):
        if isinstance(val, pd.DataFrame) or isinstance(val, pd.Series):
            val = val.to_numpy()
        return val

    def fit(self, X: np.ndarray | Any, y: np.ndarray | Any):
        X, y = self._conv(X), self._conv(y)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            X_bs, y_bs = self._prepare(X, y)

            model = self._dupmodel()
            model.fit(X_bs, y_bs)
            self.estimators_.append(model)

        return self

    def predict(self, X: np.ndarray | Any) -> np.ndarray:
        X = self._conv(X)
        preds = np.column_stack([est.predict(X) for est in self.estimators_])
        return preds.mean(axis=1)


# %%
col_cat = X_raw.select_dtypes(include=["object"]).columns
col_cat = []
col_num = X_raw.select_dtypes(include=[np.number]).columns
col_all = np.concat([col_cat, col_num])

numerical_transformer = Pipeline(steps=[("scaler", StandardScaler())])

categorical_transformer = Pipeline(
    steps=[("onehot",
            OneHotEncoder(
                handle_unknown="ignore", sparse_output=False, drop="first"))])

pipe = Pipeline(steps=[
    ("preprocessor",
     ColumnTransformer(transformers=[
         ("num", numerical_transformer, col_num),
         ("cat", categorical_transformer, col_cat),
     ])),
])

col_all  # pyright: ignore[reportUnusedExpression]

# %%
X_train, X_test, y_train, y_test = train_test_split(X_raw[col_all],
                                                    y_raw,
                                                    random_state=42,
                                                    test_size=0.33)
X_train = pipe.fit_transform(X_train, y_train)
X_test = pipe.transform(X_test)
X_train.shape

# %%
# %%time
model = BaggingRegressor(estimator=LinearRegression(), random_state=0)
model.fit(X_train, y_train)
bag_sk_pred = model.predict(X_test)
roc_auc_score(y_test, cont_to_binary(bag_sk_pred))

# %%
# %%time
bag_my = MyBaggingRegressor(estimator=LinearRegression(), random_state=0)
bag_my.fit(X_train, y_train)
bag_my_pred = bag_my.predict(X_test)
roc_auc_score(y_test, cont_to_binary(bag_my_pred))


# %%
def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sum(y_true == y_pred) / len(y_true)


def precision_score(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    average: str = "binary",
                    pos_label=1) -> float | np.float32:
    if average == "binary":
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        return 0 if tp + fp == 0 else tp / (tp + fp)

    labels = np.unique(np.concatenate([y_true, y_pred]))
    precisions = []
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        precision = 0 if tp + fp == 0 else tp / (tp + fp)
        precisions.append(precision)
    if average == "macro":
        return np.mean(precisions)
    elif average == "weighted":
        support = np.array([np.sum(y_true == label) for label in labels])
        return np.average(precisions, weights=support)
    elif average == "micro":
        tp_t = 0
        fp_t = 0
        for label in labels:
            tp_t += np.sum((y_true == label) & (y_pred == label))
            fp_t += np.sum((y_true != label) & (y_pred == label))
        return 0 if tp_t + fp_t == 0 else tp_t / (tp_t + fp_t)
    else:
        raise RuntimeError(f"invalid average: '{average}'")


def recall_score(y_true: np.ndarray,
                 y_pred: np.ndarray,
                 average="binary",
                 pos_label=1) -> float | np.float32:

    if average == "binary":
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
        return 0 if tp + fn == 0 else tp / (tp + fn)

    labels = np.unique(np.concatenate([y_true, y_pred]))
    recalls = []
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        recall = 0 if tp + fn == 0 else tp / (tp + fn)
        recalls.append(recall)
    if average == "macro":
        return np.mean(recalls)
    elif average == "weighted":
        support = np.array([np.sum(y_true == label) for label in labels])
        return np.average(recalls, weights=support)
    elif average == "micro":
        tp_t = 0
        fn_t = 0
        for label in labels:
            tp_t += np.sum((y_true == label) & (y_pred == label))
            fn_t += np.sum((y_true == label) & (y_pred != label))
        return 0 if tp_t + fn_t == 0 else tp_t / (tp_t + fn_t)
    else:
        raise RuntimeError(f"invalid average: '{average}'")


def f1_score(y_true: np.ndarray,
             y_pred: np.ndarray,
             average="binary",
             pos_label=1) -> float | np.float32:
    if average == "binary" or average == "micro":
        p = precision_score(y_true,
                            y_pred,
                            average=average,
                            pos_label=pos_label)
        r = recall_score(y_true, y_pred, average=average, pos_label=pos_label)
        return 0 if p + r == 0 else 2 * (p * r) / (p + r)
    else:
        labels = np.unique(np.concatenate([y_true, y_pred]))
        f1_scores = []
        for label in labels:
            f1_scores.append(
                f1_score(y_true, y_pred, average="binary", pos_label=label))
        if average == "macro":
            return np.mean(f1_scores)
        elif average == "weighted":
            support = np.array([np.sum(y_true == label) for label in labels])
            return np.average(f1_scores, weights=support)
        else:
            raise RuntimeError(f"invalid average: '{average}'")


# %%

pred_bin = cont_to_binary(bag_my_pred)
scores = pd.DataFrame()
scores["type"] = [
    "accuracy_score", "precision_score", "recall_score", "f1_score"
]
scores["sklearn"] = [
    metrics.accuracy_score(y_test, pred_bin),
    metrics.precision_score(y_test, pred_bin),
    metrics.recall_score(y_test, pred_bin),
    metrics.f1_score(y_test, pred_bin)
]
y_test = cast(np.ndarray, y_test)
scores["my"] = [
    accuracy_score(y_test, pred_bin),
    precision_score(y_test, pred_bin),
    recall_score(y_test, pred_bin),
    f1_score(y_test, pred_bin)
]
scores  # pyright: ignore[reportUnusedExpression]

# %%
X_fin_test = pd.read_csv("./data/test_c.csv")
X_fin_test = X_fin_test.drop(columns="ID")

pipe_final = pipe
X_fin_train = pipe_final.fit_transform(X_raw[col_all])
X_fin_test = pipe_final.transform(X_fin_test)

model = MyBaggingRegressor(estimator=LinearRegression(), random_state=0)
model.fit(X_fin_train, y_raw)
fin_pred = model.predict(X_fin_test)
fin_pred = cont_to_binary(fin_pred)

fin_out = pd.DataFrame({
    "ID": np.arange(len(fin_pred)),
    "LoanApproved": fin_pred
})
fin_out.set_index("ID", inplace=True)
fin_out.to_csv("./out.csv")
fin_out.describe()
