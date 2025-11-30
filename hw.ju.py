# %% [md]
# # hw2: Решающие деревья

# *Спасибо великому курсу великого Евгения Соколова*

# %% [md]
# ### О задании

# Задание состоит из двух разделов:
# 1. В первом разделе вы научитесь применять деревья из sklearn для задачи классификации. Вы посмотрите какие разделяющие поверхности деревья строят для различных датасетов и проанализируете их зависимость от различных гиперпараметров.
# 2. Во втором разделе вы попробуете реализовать свое
# решающее дерево и сравните его со стандартное
# имплементацией из sklearn. Вы также протестируете
# деревья на более сложных датасетах и сравните различные
# подходы к кодированию категориальных признаков.

# Все данные, на которых будут обучаться модели, вы можете найти на диске.

# ### Оценивание и штрафы
# Каждая из задач имеет определенную «стоимость»
# (указана в скобках около задачи).
# Максимально допустимая оценка за работу — 10 баллов.
# Неэффективная и/или неоригинальная реализация кода
# может негативно отразиться на оценке.

# ### Формат сдачи
# Заполненный ноутбук ```hw2-trees.ipynb``` и модуль с
# реализованными функциями и классами ```hw2code.py```
# необходимо загрузить на свой Github. Затем нужно
# оставить комментарий в Google-таблице с оценками в
# столбце <<hw2>> в строке со своей фамилией о том, что
# вы выполнили работу с указанием ника на Kaggle.

# %%
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_moons, make_circles, make_classification
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import Colormap, ListedColormap
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import accuracy_score
sns.set(style='whitegrid')

warnings.filterwarnings('ignore')

# %% [md] # # 1. Решающие деревья. Визуализация.

# %% [md]
# В этой части мы рассмотрим два простых двумерных
# датасета сделанных с помощью `make_moons`,
# `make_circles` и посмотрим как ведет себя
# разделяющая поверхность в зависимости
# от различных гиперпараметров.

# %%
datasets = [
    make_circles(noise=0.2, factor=0.5, random_state=42),
    make_moons(noise=0.2, random_state=42),
    make_classification(n_classes=3, n_clusters_per_class=1, n_features=2, class_sep=.8, random_state=3,
                        n_redundant=0, )
]

# %%
palette = sns.color_palette(n_colors=3)
cmap = ListedColormap(palette)

# %%
plt.figure(figsize=(15, 4))
for i, (x, y) in enumerate(datasets):
    plt.subplot(1, 3, i + 1)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap, alpha=.8)


# %% [md]
# __Задание 1. (1 балл)__

# Для каждого датасета обучите решающее дерево
# с параметрами по умолчанию, предварительно разбив
# выборку на обучающую и тестовую. Постройте
# разделящие поверхности (для этого воспользуйтесь
# функцией `plot_surface`, пример ниже). Посчитайте
# accuracy на обучающей и тестовой выборках.
# Сильно ли деревья переобучились?

# %%
def plot_surface(clf, X, y):
    plot_step = 0.01
    palette = sns.color_palette(n_colors=len(np.unique(y)))
    cmap = ListedColormap(palette)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.3)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, alpha=.7,
                edgecolors=np.array(palette)[y], linewidths=2)


# %%
# Пример:
X, y = datasets[2]
lr = LinearRegression().fit(X, y)
plot_surface(lr, X, y)

# %%

plt.figure(figsize=(10, 10))
trained_models_data = []
for i, (X, y) in enumerate(datasets):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=0)

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)

    train_preds = clf.predict(X_train)
    test_preds = clf.predict(X_test)
    train_res = accuracy_score(y_train, train_preds)
    test_res = accuracy_score(y_test, test_preds)

    print(f"dataset {i}:")
    print(f"\ttraining {train_res:.4f}")
    print(f"\ttesting  {test_res:.4f}")
    
    plt.subplot(3, 2, 2 * i + 1)
    plt.title(f'train dataset {i}')
    plot_surface(clf, X_train, y_train)

    plt.subplot(3, 2, 2*i + 2)
    plt.title(f'test dataset {i}')
    plot_surface(clf, X_test, y_test)


plt.tight_layout()
plt.show()


# %% [md]
# __Ответ:__ деревья сильро переобучились

# %% [md]
# __Задание 2. (1.5 балла)__

# Попробуйте перебрать несколько параметров для
# регуляризации (напр. `max_depth`, `min_samples_leaf`).
# Для каждого набора гиперпараметров постройте
# разделяющую поверхность, выведите обучающую и
# тестовую ошибки. Можно делать кросс-валидацию
# или просто разбиение на трейн и тест, главное
# делайте каждый раз одинаковое разбиение, чтобы
# можно было корректно сравнивать (помните же, что
# итоговое дерево сильно зависит от небольшого
# изменения обучающей выборки?). Проследите как
# меняется разделяющая поверхность и обобщающая
# способность. Почему так происходит, одинаково ли
# изменение для разных датасетов?

# %%
from itertools import product
max_depths = [None, 1, 3, 5]
min_samples_leaf = [1, 2, 3]
colors = ["lightcoral", "turquoise", "plum"]

dataset_cnt = len(max_depths) * len(min_samples_leaf)
for i, (X, y) in enumerate(datasets):
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=0)
    fig = plt.figure(figsize=(10, 3 * dataset_cnt))
    fig.set_facecolor(color=colors[i])

    for j, (d, l) in enumerate(product(max_depths, min_samples_leaf)):
        clf = DecisionTreeClassifier(random_state=0,max_depth=d, min_samples_leaf=l)
        clf.fit(X_train, y_train)
        
        train_preds = clf.predict(X_train)
        test_preds = clf.predict(X_test)
        train_res = accuracy_score(y_train, train_preds)
        test_res = accuracy_score(y_test, test_preds)

        print(f"dataset {i}: max_depth = {d}, min_samples_leaf = {l}")
        print(f"\ttraining {train_res:.4f}")
        print(f"\ttesting  {test_res:.4f}")
        
        plt.subplot(dataset_cnt, 2, 2 * j + 1)
        plt.title(f'train {i} depth = {d}, leaf = {l}')
        plot_surface(clf, X_train, y_train)

        plt.subplot(dataset_cnt, 2, 2*j + 2)
        plt.title(f'test {i} depth = {d}, leaf = {l}')
        plot_surface(clf, X_test, y_test)



# %% [md]
# __Ответ:__ чем больше max_depths и чем меньше min_samples_leaf тем более сложными
# являются решающие границы, что означает большую регулизацию. Из нее следуюет меньшая 
# точность при обучении, и меньший шанс переобучения, что может привести к лучшей точности
# на тестовой выборке

# %% [md]
# # 2. Решающие деревья своими руками

# %% [md]
# В этой части вам нужно реализовать свой класс
# для обучения решающего дерева в задаче бинарной
# классификации с возможностью обработки вещественных
# и категориальных признаков.

# %% [md]
# __Задание 3. (1.5 балл)__

# Реализуйте функцию find_best_split из модуля hw2code.py
# Под критерием Джини здесь подразумевается следующая функция:
# $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$
# $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
#     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.
#   Указания:
#  * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
# * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
# * Поведение функции в случае константного признака может быть любым.
# * При одинаковых приростах Джини нужно выбирать минимальный сплит.
# * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

# :param feature_vector: вещественнозначный вектор значений признака
# :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

# :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
#  разделить на две различные подвыборки, или поддерева
# :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
# :return threshold_best: оптимальный порог (число)
# :return gini_best: оптимальное значение критерия Джини (число)

# %% [md]
# __Задание 4. (0.5 балла)__

# Загрузите таблицу students.csv (это немного преобразованный датасет
# [User Knowledge](https://archive.ics.uci.edu/ml/datasets/User+Knowledge+Modeling)).
# В ней признаки объекта записаны в первых пяти
# столбцах, а в последнем записана целевая
# переменная (класс: 0 или 1). Постройте на одном
# изображении пять кривых "порог — значение критерия
# Джини" для всех пяти признаков. Отдельно визуализируйте scatter-графики "значение признака — класс"
# для всех пяти признаков.

# %%
df = pd.read_csv("./datasets/students.csv")
df

# %%
X = df.iloc[:, :5]
y = df.iloc[:, 5]

def gini_impurity(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for g in groups:
        size = len(g)
        score = sum([((g == c).sum() / size) ** 2 for c in classes])
        gini += (1.0 - score) * (size / n_instances)
    return gini


plt.figure(figsize=(10, 5))

for feature in X.columns:
    vals = X[feature].values
    thresholds = np.sort(np.unique(vals))
    impurities = [gini_impurity([y[vals <= t], y[vals > t]], [0, 1]) for t in thresholds]
    plt.plot(thresholds, impurities, label=feature)

plt.xlabel('threshold')
plt.ylabel('impurity')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 10))

for i, feature in enumerate(X.columns):
    plt.subplot(3, 2, i + 1)
    plt.scatter(X[feature], y)
    plt.title(f'{feature}')
    plt.grid(True)

plt.tight_layout()
plt.show()


# %% [md]
# __Задание 5. (0.5 балла)__

# Исходя из кривых значений критерия Джини, по какому
# признаку нужно производить деление выборки на два
# поддерева? Согласуется ли этот результат с визуальной
# оценкой scatter-графиков? Как бы охарактеризовали вид
# кривой для "хороших" признаков, по которым выборка
# делится почти идеально? Чем отличаются кривые для
# признаков, по которым деление практически невозможно?

# %% [md]
# **Ответ:** Исходя из кривых деление стоит производить по признаку PEG.
# Этот результат согласуется со scatter-шрафиками
# Чем ниже минимум gini, тем лучше этот признак подходит для разбиения.
# Кривая для хорошего признака дольжна иметь резкое падение. Кривые по которым
# деление практически невозможно гораздо более ровные, не имеют резких падений

# %% [md]
# __Задание 6. (1.5 балла).__

# Разберитесь с уже написанным кодом в классе DecisionTree модуля hw2code.py. Найдите ошибки в реализации метода
# \_fit_node. Напишите функцию \_predict_node.

#  Построение дерева осуществляется согласно базовому
# жадному алгоритму, предложенному в лекции. Выбор
# лучшего разбиения необходимо производить по критерию
# Джини. Критерий останова: все объекты в листе относятся
# к одному классу или ни по одному признаку нельзя разбить
# выборку. Ответ в листе: наиболее часто встречающийся
# класс в листе. Для категориальных признаков выполняется
# преобразование, описанное в лекции в разделе
# «Учет категориальных признаков».

# %% [md]
# __Задание 7. (0.5 балла)__

# Протестируйте свое решающее дерево на датасете
# [mushrooms](https://archive.ics.uci.edu/ml/datasets/Mushroom).
# Вам нужно скачать таблицу agaricus-lepiota.data
# (лежит на гитхабе вместе с заданием), прочитать ее
# с помощью pandas, применить к каждому столбцу
# LabelEncoder (из sklearn), чтобы преобразовать строковые
# имена категорий в натуральные числа. Первый столбец — это
# целевая переменная (e — edible, p — poisonous) Мы будем
# измерять качество с помощью accuracy, так что нам не
# очень важно, что будет классом 1, а что — классом 0.
# Обучите решающее дерево на половине случайно выбранных
# объектов (признаки в датасете категориальные) и сделайте
# предсказания для оставшейся половины. Вычислите accuracy.

# У вас должно получиться значение accuracy, равное
# единице (или очень близкое к единице), и не очень
# глубокое дерево.

# %%
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('./datasets/agaricus-lepiota.data')
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])
df

# %%
import hw2code
from importlib import reload
reload(hw2code)

X = df.iloc[:, 1:]
y = df.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0)
feature_types = ['categorical'] * X.shape[1]
X_train_vals = X_train.reset_index(drop=True).values
X_test_vals = X_test.reset_index(drop=True).values
y_train_vals = y_train.reset_index(drop=True).values
y_test_vals = y_test.reset_index(drop=True).values

tree = hw2code.DecisionTree(feature_types=feature_types)
tree.fit(X_train_vals, y_train_vals)
y_pred = tree.predict(X_test_vals)
accuracy_score(y_test_vals, y_pred)



# %% [md]
# __Задание 8. (бонус, 1 балл)__

# Реализуйте в классе DecisionTree поддержку параметров
# max_depth, min_samples_split и min_samples_leaf по
# аналогии с DecisionTreeClassifier. Постройте графики
# зависимости качества предсказания в зависимости от этих
# параметров для набора данных tic-tac-toe (см. следующий
# пункт).

# %% [md]
# __Задание 9. (2 балла)__

# Загрузите следующие наборы данных
# (напомним, что pandas умеет загружать файлы по url,
# в нашем случае это файл \*.data), предварительно
# ознакомившись с описанием признаков и целевой переменной
# в каждом из них (она записаны в Data Folder, в файле
# *.names):
# * [mushrooms](https://archive.ics.uci.edu/ml/datasets/Mushroom)
# (загрузили в предыдущем пункте, классы записаны в нулевом столбце)
# * [tic-tac-toe](https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame)
# (классы записаны в последнем столбце, датасет лежит на
# гитхабе вместе с заданием)
# * [cars](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation)
# (классы записаны в последнем столбце, считаем
# что unacc, acc — это класс 0, good, vgood — класс 1)
# * [nursery](https://archive.ics.uci.edu/ml/datasets/Nursery)
# (классы записаны в последнем столбце, считаем, что
# not_recom и recommend — класс 0, very_recom, priority,
# spec_prior — класс 1).

# Закодируйте категориальные признаки, использовав LabelEncoder. С помощью cross_val_score (cv=10) оцените accuracy на каждом из этих наборов данных следующих алгоритмов:
# * DecisionTree, считающий все признаки вещественными
# * DecisionTree, считающий все признаки категориальными
# * DecisionTree, считающий все признаки вещественными
# + one-hot-encoding всех признаков
# * DecisionTreeClassifier из sklearn. Запишите результат
# в pd.DataFrame (по строкам — наборы данных, по
# столбцам — алгоритмы).

# Рекомендации:
# * Чтобы cross_val_score вычисляла точность, нужно
# передать scoring=make_scorer(accuracy_score),
# обе фукнции из sklearn.metrics.
# * Если вам позволяет память (а она скорее всего
# позволяет), указывайте параметр sparse=False в
# OneHotEncoder (если вы, конечно, используете его).
# Иначе вам придется добиваться того, чтобы ваша
# реализация дерева умела работать с разреженными
# матрицами (что тоже, в целом, не очень сложно).

# %%
# ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ


# %% [md]
# __Задание 10. (1 балла)__

# Проанализируйте результаты эксперимента.
# Одинаково ли для разных наборов данных ранжируются алгоритмы?
# Порассуждайте, почему так происходит.

# Обратите внимание на значение признаков в разных наборах данных.
# Присутствует ли в результатах какая-то компонента случайности?
# Можно ли повлиять на нее и улушить работу алгоритмов?

# %% [md]
# **Ответ:**

# %% [md]
# Вставьте что угодно, описывающее ваши впечатления от этого задания:

# %%
