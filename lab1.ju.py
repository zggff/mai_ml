# %% [md]
# # Лабораторная работа №1 | Вариант 5

# %% [md]
# Выполнил Гига Максим М8О-313Б-23

# %% [md]
# ### 1. Numpy


# %%
import numpy as np
from numpy.typing import NDArray


# %% [md]
# 1. Создайте единичную матрицу размером 3х3

# %%
np.identity(3)

# %% [md]
# 2. Создайте кастомный dtype который описывает цвет
# состоящий из 4х unsigned bytes(RGBA пиксель)

# %%
np.dtype([("r", np.uint8), ("g", np.uint8), ("b", np.uint8), ("a", np.uint8)])

# %% [md]
# 3. Как игнорировать все numpy warnings(не рекомендуем к использованию)?

# %%
# np.seterr(all="ignore")


# %% [md]
# 4. Создайте случайный вектор размера 10 и отсортируйте его

# %%
arr = np.random.rand(1, 10)
arr.sort()
arr                 # type:ignore

# %% [md]
# 5. Создайте read-only массив(неизменяемый)

# %%
read_only = np.zeros(10)
read_only.setflags(write=False)

# %% [md]
# 6. Рассмотрим случайную матрицу размером 10х2,
# представляющую декартовы координаты, преобразуем их в полярные
# координаты


# %%
def cartesian_to_polar(xs: NDArray[np.float64]):
    x = xs[:, 0]
    y = xs[:, 1]
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return np.stack((rho, phi), axis=1)


xy = np.random.random((10, 2))
cartesian_to_polar(xy)

# %% [md]
# 7. Создайте структурированный массив с координатами x и y,
# охватывающий область [0,1]x[0,1]

# %%
np.mgrid[0:1.01:0.01, 0:1.01:0.01].reshape(2, -1).T

# %% [md]
# 8. Учитывая два массива X и Y, постройте матрицу
# Коши $C$ ($C_{ij} =\frac{1}{x_i - y_j}$)

# %%

x = np.random.normal(size=(5))
y = np.random.normal(size=(5))

c = np.fromfunction(lambda i, j: 1/(x[i] + y[j]), shape=(5, 5), dtype=int)
c                   # type: ignore

# %% [md]
# 9. Как конвертировать массив float(32 bit) к массиву целых чисел
# (integer 32 bit) in place?

# %%
x = np.random.normal(scale=10, size=(1, 10))
x = x.astype(int)
x                   # type: ignore

# %% [md]
# 10. Как случайно заменить p элементов в 2D массиве?


# %%
def change_random(arr: NDArray, new_val, p: int):
    arr[np.random.choice(arr.size, p, replace=False)] = new_val


x = np.zeros(10)
change_random(x, 1, 4)
x               # type: ignore


# %% [md]
# ### 2. Pandas

# %% [md]
# Найдите ответы на вопросы или выполните действия с предложенным датасетом

# %% [md]
# Классический начальный набор данных - данные пассажиров титаника.

# Файл: `data.csv`

# ```
# PassangerId - уникальный номер пассажира
# Survived - выжил человек или нет
# Pclass - класс обслуживания
# Name - имя человека, обращение к ней/нему
# Sex - пол человека
# Age - возраст на момент крушения
# SibSp - количество братьев и сестер / супругов на борту "Титаника"
# Parch - количество родителей / детей на борту "Титаника"
# Ticket - номер билета
# Fare - стоимость проезда для пассажиров
# Cabin - каюта
# Embarked - причал отбытия. C = Cherbourg Q = Queenstown S = Southampton
# ```

# %%
# Load dataset here
import pandas as pd  # noqa: E402
df = pd.read_csv("./data.csv")

# %% [md]
# 1. Узнайте сколько было на борту человек

# %%

len(df)


# %% [md]
# 2. Какой медианный и средний возраста пассажиров

# %%

print(df["Age"].median())
print(df["Age"].mean())


# %% [md]
# 3. Посчитайте процент выживаемости детей(до 16 лет) и взрослых

# %%

ch = df[df["Age"] <= 16]
ad = df[df["Age"] > 16]
chs = ch[ch["Survived"] == 1]
ads = ad[ad["Survived"] == 1]
print(f"children: {len(chs)/len(ch) * 100}%")
print(f"adults:   {len(ads)/len(ad) * 100}%")


# %% [md]
# 4. Верна ли гипотеза, что женщины и дети сажались в шлюпки
# первыми и выживали больше?

# %%
mask = (df["Age"] <= 16) | (df["Sex"] == 'female')
fc = df[mask]
mo = df[~mask]
print(f"woman and children: {len(fc[fc['Survived'] == 1]) / len(fc) * 100}%")
print(f"adult males:        {len(mo[mo['Survived'] == 1]) / len(mo) * 100}%")


# %% [md]
# 5. Зависит ли выживаемость от класса обслуживания?

# %%
for cl in sorted(df['Pclass'].unique()):
    dfcl = df[df['Pclass'] == cl]
    print(f"{cl}: {len(dfcl[dfcl['Survived'] == 1]) / len(dfcl) * 100}%")


# %% [md]
# 6. Посчитайте средний возраст умерших женщин и мужчин

# %%
female = df[(df["Sex"] == 'female') & (df['Survived'] == 0)]['Age'].mean()
male = df[(df["Sex"] == 'male') & (df['Survived'] == 0)]['Age'].mean()
print(f"female = {female}, male = {male}")


# %% [md]
# 7. Различается ли процент выживаемости пассажиров с хотя бы
# одним родственником на борту и одиночек?

# %%
rel = df[(df["SibSp"] > 0) | (df["Parch"] > 0)]
lon = df[(df["SibSp"] == 0) & (df["Parch"] == 0)]
print(f"with:    {len(rel[rel['Survived'] == 0])/len(rel)*100}%")
print(f"without: {len(lon[lon['Survived'] == 0])/len(lon)*100}%")


# %% [md]
# 8. Различается ли средняя стоимость билета у умерших и выживших пассажиров?

# %%
print(f"dead:  {df[df['Survived'] == 0]['Fare'].mean()}")
print(f"alive: {df[df['Survived'] == 1]['Fare'].mean()}")


# %% [md]
# 9. Выведите максимальный и минимальный возраст выживших
# пассажиров во 2 классе обслуживания

# %%
cl2 = df[df["Pclass"] == 2]["Age"]
print(f"max = {cl2.max()}, min = {cl2.min()}")


# %% [md]
# 10. Подтвердите или опровергните гипотезу: молодым
# мужчинам(от 18 лет) выжить легче, чем более взрослым

# %%

mo = df[df["Sex"] == "male"]
my = mo[mo["Age"] <= 18]
mo = mo[mo["Age"] > 18]
print(f"young: {len(my[my['Survived'] == 1]) / len(my) * 100}%")
print(f"old:   {len(mo[mo['Survived'] == 1]) / len(mo) * 100}%")

# %% [md]
# ### 3. Визуализация

# %% [md]
# Используя библиотеки matplotlib/sns/plotly/pandas и набор данных
# представленный ранее визуализируйте

# %% [md]
# 1. Постройте гистограмму распределения возростов пассажиров

# %%


# %% [md]
# 2. Постройте гистограммы распределения цен для пассажиров разных классов

# %%


# %% [md]
# 3. Постройте зависимость цены от возраста пассажира

# %%


# %% [md]
# 4. Постройте box plot отображающий распределение цен на билеты
# в разных классах

# %%


# %% [md]
# 5. Визуализируйте распределение долей выживших среди мужчин,
# женщин и детей(до 16 лет)

# %%


# %% [md]
# 6. Сравните плотности распределения возростов выживших и
# умерших пассажиров, сделайте выводы

# %%


# %% [md]
# 7. Постройте круговую диаграмму отображающую выживаемость мужчин и
# женщин в разных классах обслуживания(визуализация должна быть
# интуитивно понятная без объяснений, женщины и мужчины
# соответствующих классов объеденены визуально в одну группу
# и 2 подгруппы)*

# \* Задание на доп. балл

# %%
