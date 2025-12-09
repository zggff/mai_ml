# %% [md]
# ## Гига М.Я. М80-313Б-33

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import scipy.stats as stats

import lightgbm
import xgboost
import catboost
import optuna

from typing import Any, Tuple, cast

pd.options.display.max_columns = None

import warnings

warnings.filterwarnings("ignore")

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
if True:
    for col in df_filled.select_dtypes(include=np.number).columns:
        sns.violinplot(y=df_filled[col], x=y_raw)
        plt.title(col)
        plt.show()


# %%
class MyBaggingClassifier:

    def __init__(self,
                 estimator: Any = DecisionTreeClassifier(),
                 n_estimators: int = 10,
                 sample_size: int = 1,
                 random_state: int | None = None):
        self.est = estimator
        self.n_estimators = n_estimators
        self.sample_size = sample_size
        self.random_state = random_state

    def _prepare(self, X: np.ndarray,
                 y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = X.shape[0]
        size = n_samples * self.sample_size
        indices = np.random.choice(n_samples, size=size, replace=True)
        return X[indices], y[indices]

    def _conv(self, val: Any):
        if isinstance(val, pd.DataFrame) or isinstance(val, pd.Series):
            val = val.to_numpy()
        return val

    def fit(self, X: np.ndarray | Any, y: np.ndarray | Any):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        X, y = self._conv(X), self._conv(y)
        self.estimators = []
        for _ in range(self.n_estimators):
            X_bs, y_bs = self._prepare(X, y)

            model: Any = clone(self.est)
            model.fit(X_bs, y_bs)
            self.estimators.append(model)

        return self

    def predict(self, X: np.ndarray | Any) -> np.ndarray:
        X = self._conv(X)
        preds = np.column_stack([est.predict(X) for est in self.estimators])
        preds, _ = stats.mode(
            preds,
            axis=1,
        )
        return preds.ravel()

    def predict_proba(self, X: np.ndarray | Any) -> np.ndarray:
        X = self._conv(X)
        preds = np.column_stack(
            [est.predict_proba(X) for est in self.estimators])
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
bag_sk = BaggingClassifier(estimator=DecisionTreeClassifier(), random_state=0)
bag_sk.fit(X_train, y_train)
bag_sk_pred = bag_sk.predict(X_test)
metrics.roc_auc_score(y_test, bag_sk_pred)

# %%
# %%time
bag_my = MyBaggingClassifier(estimator=DecisionTreeClassifier(),
                             random_state=0)
bag_my.fit(X_train, y_train)
bag_my_pred = bag_my.predict(X_test)
metrics.roc_auc_score(y_test, cont_to_binary(bag_my_pred))


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


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    indices = np.argsort(y_score)[::-1]
    y_score = y_score[indices]
    y_true = y_true[indices]

    pos_i = np.where(y_true == 1)[0]
    pos_cnt = len(pos_i)
    neg_cnt = len(y_true) - pos_cnt

    ranks = np.empty(len(y_score))
    unique_scores, inverse_indices, counts = np.unique(y_score,
                                                       return_inverse=True,
                                                       return_counts=True)

    rank = 1.0
    for i in range(len(unique_scores)):
        avg_rank = rank + (counts[i] - 1) / 2.0
        ranks[inverse_indices == i] = avg_rank
        rank += counts[i]

    return (np.sum(ranks[pos_i]) - pos_cnt *
            (pos_cnt + 1) / 2.0) / (pos_cnt * neg_cnt)


def pr_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    indices = np.argsort(y_score)[::-1]
    y_true = y_true[indices]
    y_score = y_score[indices]
    thresholds = np.unique(y_score)[::-1]
    tp = fp = 0
    total_positives = np.sum(y_true == 1)

    if total_positives == 0:
        return 0.0

    precision_values = [1.0]
    recall_values = [0.0]

    for threshold in thresholds:
        pred_positives = y_score >= threshold
        tp = np.sum(y_true[pred_positives] == 1)
        fp = np.sum(y_true[pred_positives] == 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / total_positives

        precision_values.append(precision)
        recall_values.append(recall)

    tp = total_positives
    fp = len(y_true) - total_positives
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    precision_values.append(precision)
    recall_values.append(1.0)
    return np.trapezoid(precision_values, recall_values)


# %%

pred_bin = bag_my_pred
scores = pd.DataFrame()
scores["type"] = [
    "accuracy_score", "precision_score", "recall_score", "f1_score", "roc_auc",
    "pr_auc"
]
precision, recall, _ = metrics.precision_recall_curve(y_test, pred_bin)
scores["sklearn"] = [
    metrics.accuracy_score(y_test, pred_bin),
    metrics.precision_score(y_test, pred_bin),
    metrics.recall_score(y_test, pred_bin),
    metrics.f1_score(y_test, pred_bin),
    metrics.roc_auc_score(y_test, pred_bin),
    metrics.auc(recall, precision),
]
y_test = np.array(y_test)
scores["my"] = [
    accuracy_score(y_test, pred_bin),
    precision_score(y_test, pred_bin),
    recall_score(y_test, pred_bin),
    f1_score(y_test, pred_bin),
    roc_auc_score(y_test, pred_bin),
    pr_auc_score(y_test, pred_bin),
]
scores  # pyright: ignore[reportUnusedExpression]


# %%
class MyGradientBoostingClassifier:

    def __init__(self,
                 learning_rate: float = 0.1,
                 n_estimators: int = 100,
                 max_depth: int = 3,
                 min_samples_leaf: int = 1,
                 min_samples_split: int = 2,
                 random_state: float | None = None) -> None:
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.random_state = random_state

    def _conv(self, val: Any) -> np.ndarray:
        if isinstance(val, pd.DataFrame) or isinstance(val, pd.Series):
            val = val.to_numpy()
        return val

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X: np.ndarray | Any, y: np.ndarray | Any):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        X, y = self._conv(X), self._conv(y)

        pos_prob = np.mean(y)

        self.initial_prediction = np.log(pos_prob / (1 - pos_prob))
        self.trees = []

        current_predictions = np.full(len(y), self.initial_prediction)

        for _ in range(self.n_estimators):
            current_probs = self._sigmoid(current_predictions)
            residuals = y - current_probs

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state)
            tree.fit(X, residuals)

            tree_predictions = tree.predict(X)
            current_predictions += self.learning_rate * tree_predictions
            self.trees.append(tree)
        return self

    def predict(self, X: np.ndarray | Any) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray | Any) -> np.ndarray:
        X = self._conv(X)
        predictions = np.full(len(X), self.initial_prediction)

        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        prob_positive = self._sigmoid(predictions)
        return np.column_stack([1 - prob_positive, prob_positive])


# %%
# %%time
grad_sk = GradientBoostingClassifier(random_state=0)
grad_sk.fit(X_train, y_train)
grad_sk_pred = grad_sk.predict(X_test)
metrics.roc_auc_score(y_test, grad_sk_pred)

# %%
# %%time
grad_my = MyGradientBoostingClassifier(random_state=0)
grad_my.fit(X_train, y_train)
grad_my_pred = grad_my.predict(X_test)
metrics.roc_auc_score(y_test, grad_my_pred)

# %% [md]
# сравнивание реализаций
# %%
comp = pd.DataFrame()
models = [
    ("sklearn", GradientBoostingClassifier(random_state=0)),
    ("XGBoost", xgboost.XGBClassifier(random_state=0)),
    ("LightGBM", lightgbm.LGBMClassifier(random_state=0, verbose=-1)),
    ("CatBoost", catboost.CatBoostClassifier(random_state=0, verbose=0)),
]
comp["name"] = [
    "ROC_ACU", "accuracy_score", "precision_score", "recall_score", "f1_score"
]
for name, model in models:
    print(f"fit {name}")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pred = cont_to_binary(pred)
    comp[name] = [
        metrics.roc_auc_score(y_test, pred),
        metrics.accuracy_score(y_test, pred),
        metrics.precision_score(y_test, pred),
        metrics.recall_score(y_test, pred),
        metrics.f1_score(y_test, pred),
    ]

comp  # pyright: ignore[reportUnusedExpression]

# %% [md]
# модели не показывают сильного различия в результате


# %%
def objective(trial):
    params = {
        "num_iterations":
        trial.suggest_int("num_iterations", 50, 150, step=10),
        "learning_rate":
        trial.suggest_float("learning_rate", 0.05, 0.3),
        "num_leaves":
        trial.suggest_int("num_leaves", 20, 50, step=5),
        "min_child_samples":
        trial.suggest_int("min_child_samples", 20, 500, step=20),
        "random_state":
        0,
        "verbose":
        -1,
    }

    model = lightgbm.LGBMClassifier(**params)
    # Используем cross-validation для оценки
    scores = cross_val_score(model,
                             X_raw[col_all],
                             y_raw,
                             cv=3,
                             scoring='roc_auc',
                             n_jobs=-1)
    return scores.mean()


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=4)
study.best_params.items()

# %%
X_fin_test = pd.read_csv("./data/test_c.csv")
X_fin_test = X_fin_test.drop(columns="ID")

pipe_final = pipe
X_fin_train = pipe_final.fit_transform(X_raw[col_all])
X_fin_test = pipe_final.transform(X_fin_test)

model = MyGradientBoostingClassifier(random_state=0)
model.fit(X_fin_train, y_raw)
fin_pred = np.array(model.predict(X_fin_test))

fin_out = pd.DataFrame({
    "ID": np.arange(len(fin_pred)),
    "LoanApproved": fin_pred
})
fin_out.set_index("ID", inplace=True)
fin_out.to_csv("./out.csv")
fin_out.describe()
