# %% [md]
# # hw3: Обучение без учителя

# *Спасибо ещё одному великому курсу mlcourse.ai и авторам: Ольга Дайховская (@aiho в Slack ODS), Юрий Кашницкий (@yorko в Slack ODS).*

# %% [md]
"""
### О задании
В этом задании мы разберемся с тем, как работают методы снижения
размерности и кластеризации данных. Заодно еще раз попрактикуемся в
задаче классификации.

Мы будем работать с набором данных
[Samsung Human Activity Recognition](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones).
Данные поступают с акселерометров и гироскопов мобильных телефонов
Samsung Galaxy S3 (подробнее про признаки – по ссылке на UCI выше),
также известен вид активности человека с телефоном в кармане –
ходил ли он, стоял, лежал, сидел или шел вверх/вниз по лестнице.

Вначале мы представим, что вид активности нам неизвестнен, и
попробуем кластеризовать людей чисто на основе имеющихся признаков.
Затем решим задачу определения вида физической активности именно как
задачу классификации.

**Заполните код в клетках (где написано "Ваш код здесь") и ответьте
на вопросы, выделив ответ полужирным** (``` **выделить двойными
звёздочками** ```).

### Оценивание и штрафы
Вам необходимо ответить на 10 вопросов и выполнить 2 задания.
Каждое из заданий и вопросов имеет определенную «стоимость»
(указана в скобках). Максимально допустимая оценка за работу — 10
баллов. Неэффективная и/или неоригинальная реализация кода может
негативно отразиться на оценке.

### Формат сдачи
Заполненный ноутбук ```hw3-unsupervised.ipynb``` необходимо загрузить
на свой Github. Затем нужно оставить комментарий в Google-таблице с
оценками в столбце "hw3" в строке со своей фамилией о том, что вы
выполнили работу и оставить ссылку на ноутбук.
"""  # noqa 501


# %%
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn import metrics
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm_notebook

# %matplotlib inline
from matplotlib import pyplot as plt

plt.style.use(['seaborn-v0_8-darkgrid'])
plt.rcParams['figure.figsize'] = (12, 9)
plt.rcParams['font.family'] = 'DejaVu Sans'


RANDOM_STATE = 17

# %%
X_train = np.loadtxt("./datasets/UCI HAR Dataset/train/X_train.txt")
y_train = np.loadtxt(
    "./datasets/UCI HAR Dataset/train/y_train.txt").astype(int)

X_test = np.loadtxt("./datasets/UCI HAR Dataset/test/X_test.txt")
y_test = np.loadtxt("./datasets/UCI HAR Dataset/test/y_test.txt").astype(int)

# %%
# Проверим размерности
assert (X_train.shape == (7352, 561) and y_train.shape == (7352,))
assert (X_test.shape == (2947, 561) and y_test.shape == (2947,))

# %% [md]
"""
Для кластеризации нам не нужен вектор ответов, поэтому будем
работать с объединением обучающей и тестовой выборок.
Объедините *X_train* с *X_test*, а *y_train* – с *y_test*.
"""

# %%
X = np.concatenate([X_train, X_test])
y = np.concatenate([y_train, y_test])

# %% [md]
"""
Определим число уникальных значений меток целевого класса.
"""

# %%
np.unique(y)

# %%
n_classes = np.unique(y).size

# %% [md]
"""
[Эти метки соответствуют:](https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.names)
 - 1 - ходьбе
 - 2 - подъему вверх по лестнице
 - 3 - спуску по лестнице
 - 4 - сидению
 - 5 - стоянию
 - 6 - лежанию
*уж простите, если звучание этих существительных кажется корявым :)*
"""  # noqa 501

# %% [md]
"""
Отмасштабируйте выборку с помощью `StandardScaler` с
параметрами по умолчанию.
"""


# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %% [md]
"""Понижаем размерность с помощью PCA, оставляя столько компонент, сколько нужно для того, чтобы объяснить как минимум 90% дисперсии исходных (отмасштабированных) данных. Используйте отмасштабированную выборку и зафиксируйте random_state (константа RANDOM_STATE)."""  # noqa 501

# %%
# Ваш код здесь
pca = PCA(random_state=RANDOM_STATE, n_components=0.9)
X_pca = pca.fit_transform(X_scaled)

# %% [md]
"""
**Вопрос 1:** (1 балл)

Какое минимальное число главных компонент нужно выделить,
чтобы объяснить 90% дисперсии исходных (отмасштабированных) данных?

**Варианты:**
- 56
- **65**
- 66
- 193
"""

# %%
print(X_pca.shape[1])

# %% [md]
"""
**Вопрос 2:** (0.5 баллов)

Сколько процентов дисперсии приходится на первую главную компоненту?
Округлите до целых процентов.

**Варианты:**
- 45
- **51**
- 56
- 61
"""


# %%
print(int(round(pca.explained_variance_ratio_[0], 2) * 100))

# %% [md]
"""
Визуализируйте данные в проекции на первые две главные компоненты.
"""

# %%
num_to_activity = {
    1:  "ходьбе",
    2:  "подъему вверх по лестнице",
    3:  "спуску по лестнице",
    4:  "сидению",
    5:  "стоянию",
    6:  "лежанию"
}
scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1], c=y, s=20, cmap='viridis')
lines, vals = scatter.legend_elements()
labels = [num_to_activity[int(v[-3])] for v in vals]
plt.legend(lines, labels)
plt.show()

# %% [md]
"""
**Вопрос 3:** (0.5 баллов)

Если все получилось правильно, Вы увидите сколько-то кластеров, 
почти идеально отделенных друг от друга. Какие виды активности входят в эти кластеры?<br>

**Ответ:**
- 1 кластер: все 6 активностей
- **2 кластера: (ходьба, подъем вверх по лестнице, спуск по лестнице) и (сидение, стояние, лежание)**
- 3 кластера: (ходьба), (подъем вверх по лестнице, спуск по лестнице) и (сидение, стояние, лежание)
- 6 кластеров
"""  # noqa 501

# %% [md]
"""
**Задание 1.** (1 балл)

Сделайте кластеризацию данных методом `KMeans` (собственная имплементация и готовая реализация), обучив модель на данных со сниженной за счет PCA размерностью. В данном случае мы подскажем, что нужно искать именно 6 кластеров, но в общем случае мы не будем знать, сколько кластеров надо искать.

Параметры:

- **n_clusters** = n_classes (число уникальных меток целевого класса)
- **n_init** = 100
- **random_state** = RANDOM_STATE (для воспроизводимости результата)

Остальные параметры со значениями по умолчанию.
"""  # noqa 501


# %%
class MyKMeans:
    def __init__(self, n_clusters: int, n_init: int = 100,
                 max_iter: int = 300, tol=1e-4,
                 random_state: int | None = None):
        self.random_state = random_state
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        random_indices = np.random.choice(
            n_samples, self.n_clusters, replace=False)
        return X[random_indices]

    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        distances = np.zeros((X.shape[0], self.n_clusters))

        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)

        # Assign each point to the nearest centroid
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X: np.ndarray, labels):
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))

        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                new_centroids[i] = X[np.random.randint(0, X.shape[0])]

        return new_centroids

    def _calculate_inertia(self, X: np.ndarray, labels) -> float:
        inertia = 0
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum(np.linalg.norm(cluster_points -
                                  self.centroids[i], axis=1) ** 2)
        return inertia

    def fit(self, X):
        X = np.array(X)

        self.centroids = self._initialize_centroids(X)

        for iteration in range(self.max_iter):
            old_centroids = self.centroids.copy()
            self.labels = self._assign_clusters(X)
            self.centroids = self._update_centroids(X, self.labels)
            self.inertia_ = self._calculate_inertia(X, self.labels)
            centroid_shift = np.linalg.norm(self.centroids - old_centroids)
            if centroid_shift < self.tol:
                break
        return self


kmeans_sk = KMeans(
    n_clusters=n_classes,
    n_init=100,  # pyright: ignore
    random_state=RANDOM_STATE)
kmeans_sk.fit(X_pca)

kmeans_my = MyKMeans(
    n_clusters=n_classes,
    n_init=100,  # pyright: ignore
    random_state=RANDOM_STATE)
kmeans_my.fit(X_pca)

cluster_labels = kmeans_sk.labels_
print(np.bincount(kmeans_sk.labels_))
print(np.bincount(kmeans_my.labels))


# %% [md]
"""
Визуализируйте данные в проекции на первые две главные компоненты. Раскрасьте точки в соответствии с полученными метками кластеров.
"""  # noqa 501

# %%
scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1], c=kmeans_sk.labels_, s=20, cmap='viridis')
lines, vals = scatter.legend_elements()
labels = [num_to_activity[int(v[-3]) + 1] for v in vals]
plt.legend(lines, labels)
plt.show()

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1], c=kmeans_my.labels, s=20, cmap='viridis')
lines, vals = scatter.legend_elements()
labels = [num_to_activity[int(v[-3]) + 1] for v in vals]
plt.legend(lines, labels)
plt.show()


# %% [md]
"""
Посмотрите на соответствие между метками кластеров и исходными метками
классов и на то, какие виды активностей алгоритм `KMeans` путает.
"""

# %%
tab = pd.crosstab(y, cluster_labels, margins=True)
tab.index = ['ходьба', 'подъем вверх по лестнице',
             'спуск по лестнице', 'сидение', 'стояние', 'лежание', 'все']
tab.columns = ['cluster' + str(i + 1) for i in range(6)] + ['все']
tab

# %% [md]
"""
Видим, что каждому классу (т.е. каждой активности) соответствуют несколько кластеров. Давайте посмотрим на максимальную долю объектов в классе, отнесенных к какому-то одному кластеру. Это будет простой метрикой, характеризующей, насколько легко класс отделяется от других при кластеризации.

Пример: если для класса "спуск по лестнице", в котором 1406 объектов,  распределение кластеров такое:
- кластер 1 – 900
- кластер 3 – 500
- кластер 6 – 6,

то такая доля будет 900 / 1406 $\approx$ 0.64.
"""  # noqa 501

# %% [md]
"""
**Вопрос 4:** (1 балл)

Какой вид активности отделился от остальных лучше всего в терминах простой  метрики, описанной выше?<br>

**Ответ:**
- ходьба
- стояние
- спуск по лестнице
- **перечисленные варианты не подходят**
"""  # noqa 501

# %% [md]
"""
Видно, что kMeans не очень хорошо отличает только активности друг от друга. Используйте метод локтя, чтобы выбрать оптимальное количество кластеров. Параметры алгоритма и данные используем те же, что раньше, меняем только `n_clusters`.
"""  # noqa 501

# %%
# Ваш код здесь
inertia = []
for k in tqdm_notebook(range(1, n_classes + 1)):
    pass

# %% [md]
"""
**Вопрос 5:** (1 балл)

Какое количество кластеров оптимально выбрать, согласно методу локтя?<br>

**Ответ:**
- 1
- 2
- 3
- 4
"""

# %% [md]
"""
Попробуем еще один метод кластеризации, который описывался в статье
– агломеративную кластеризацию.
"""

# %%
ag = AgglomerativeClustering(n_clusters=n_classes,
                             linkage='ward').fit(X_pca)

# %% [md]
"""
Посчитайте Adjusted Rand Index (`sklearn.metrics`)
для получившегося разбиения на кластеры и для `KMeans` с
параметрами из задания к 4 вопросу.

"""

# %%
# Ваш код здесь

# %% [md]
"""
**Вопрос 6:** (1 балл)

Отметьте все верные утверждения.<br>

**Варианты:**
- Согласно ARI, KMeans справился с кластеризацией хуже, чем Agglomerative Clustering
- Для ARI не имеет значения какие именно метки присвоены кластерам, имеет значение только разбиение объектов на кластеры
- В случае случайного разбиения на кластеры ARI будет близок к нулю
"""

# %% [md]
# -------------------------------

# %% [md]
# Можно заметить, что задача не очень хорошо решается именно как задача кластеризации, если выделять несколько кластеров (> 2). Давайте теперь решим задачу классификации, вспомнив, что данные у нас размечены.

# Для классификации используйте метод опорных векторов – класс `sklearn.svm.LinearSVC`. Мы в курсе отдельно не рассматривали этот алгоритм, но он очень известен, почитать про него можно, например, в материалах Евгения Соколова –  [тут](https://github.com/esokolov/ml-course-msu/blob/master/ML16/lecture-notes/Sem11_linear.pdf).

# Настройте для `LinearSVC` гиперпараметр `C` с помощью `GridSearchCV`.

# - Обучите новый `StandardScaler` на обучающей выборке (со всеми исходными признаками), прмиените масштабирование к тестовой выборке
# - В `GridSearchCV` укажите  cv=3.

# %%
# Ваш код здесь
#
X_train_scaled = None
X_test_scaled = None

# %%
svc = LinearSVC(random_state=RANDOM_STATE)
svc_params = {'C': [0.001, 0.01, 0.1, 1, 10]}

# %%
# Ваш код здесь
best_svc = svc

# %%
# Ваш код здесь
pass

# %% [md]
# **Вопрос 7** (0.5 баллов)

# Какое значение гиперпараметра `C` было выбрано лучшим по итогам кросс-валидации?<br>

# **Ответ:**
# - 0.001
# - 0.01
# - 0.1
# - 1
# - 10

# %%
y_predicted = best_svc.predict(X_test_scaled)

# %%
tab = pd.crosstab(y_test, y_predicted, margins=True)
tab.index = ['ходьба', 'подъем вверх по лестнице', 'спуск по лестнице',
             'сидение', 'стояние', 'лежание', 'все']
tab.columns = tab.index
tab

# %% [md]
# **Вопрос 8:** (0.5 балл)

# Какой вид активности SVM определяет хуже всего в терминах точности? Полноты? <br>

# **Ответ:**
# - по точности – подъем вверх по лестнице, по полноте – лежание
# - по точности – лежание, по полноте – сидение
# - по точности – ходьба, по полноте – ходьба
# - по точности – сидение, по полноте – стояние

# %% [md]
# Наконец, проделайте то же самое, что в 7 вопросе, только добавив PCA.

# - Используйте выборки `X_train_scaled` и `X_test_scaled`
# - Обучите тот же PCA, что раньше, на отмасшабированной обучающей выборке, примените преобразование к тестовой
# - Настройте гиперпараметр `C` на кросс-валидации по обучающей выборке с PCA-преобразованием. Вы заметите, насколько это проходит быстрее, чем раньше.

# **Вопрос 9:** (1 балл)

# Какова разность между лучшим качеством (долей верных ответов) на кросс-валидации в случае всех 561 исходных признаков и во втором случае, когда применялся метод главных компонент? Округлите до целых процентов.<br>

# **Варианты:**
# - Качество одинаковое
# - 2%
# - 4%
# - 10%
# - 20%


# %% [md]
# **Вопрос 10:** (1 балл)

# Выберите все верные утверждения:

# **Варианты:**
# - Метод главных компонент в данном случае позволил уменьшить время обучения модели, при этом качество (доля верных ответов на кросс-валидации) очень пострадало, более чем на 10%
# - PCA можно использовать для визуализации данных, однако для этой задачи есть и лучше подходящие методы, например, tSNE. Зато PCA имеет меньшую вычислительную сложность
# - PCA строит линейные комбинации исходных признаков, и в некоторых задачах они могут плохо интерпретироваться человеком

# %% [md]
# **Задание 2.** (1 балл)

# Попробуйте использовать DBSCAN в качестве алгоритма кластеризации и метод понижения размерности tSNE.
