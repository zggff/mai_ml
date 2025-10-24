# %% [md]
# # Лабораторная работа №1 | Вариант 5

# %% [md]
# Выполнил Гига Максим М8О-313Б-23

# %% [md]
# ### 1. Numpy


# %%
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
arr = np.random.rand(10)
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

print(x, y)
c = np.fromfunction(lambda i, j: 1/(x[i] + y[j]), shape=(5, 5), dtype=int)
c                   # type: ignore

# %% [md]
# 9. Как конвертировать массив float(32 bit) к массиву целых чисел
# (integer 32 bit) in place?

# %% convert values to type
x = np.random.normal(scale=10, size=5)
x = x.astype(int)   # this converts values, result is the original rounded
x                   # type: ignore

# %% in place
x = np.random.normal(scale=10, size=5)
x = x.view(int)     # this converts types
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
df = pd.read_csv("./data.csv")

# %% [md]
# 1. Узнайте сколько было на борту человек

# %%

len(df)


# %% [md]
# 2. Какой медианный и средний возраста пассажиров

# %%

print("медианный =", df["Age"].median())
print("средний =  ", df["Age"].mean())


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
# Гипотеза верна

# %% [md]
# 5. Зависит ли выживаемость от класса обслуживания?

# %%
for cl in sorted(df['Pclass'].unique()):
    dfcl = df[df['Pclass'] == cl]
    print(f"{cl}: {len(dfcl[dfcl['Survived'] == 1]) / len(dfcl) * 100}%")

# %% [md]
# Да, зависит

# %% [md]
# 6. Посчитайте средний возраст умерших женщин и мужчин

# %%
fema = df[(df["Sex"] == 'female') & (df['Survived'] == 0)]['Age'].mean()
male = df[(df["Sex"] == 'male') & (df['Survived'] == 0)]['Age'].mean()
print(f"female = {fema}, male = {male}")


# %% [md]
# 7. Различается ли процент выживаемости пассажиров с хотя бы
# одним родственником на борту и одиночек?

# %%
rel = df[(df["SibSp"] > 0) | (df["Parch"] > 0)]
lon = df[(df["SibSp"] == 0) & (df["Parch"] == 0)]
print(f"with:    {len(rel[rel['Survived'] == 0])/len(rel)*100}%")
print(f"without: {len(lon[lon['Survived'] == 0])/len(lon)*100}%")

# %% [md]
# Различаются. Люди с родственниками имели меньший шанс выживания


# %% [md]
# 8. Различается ли средняя стоимость билета у умерших и выживших пассажиров?

# %%
print(f"dead:  {df[df['Survived'] == 0]['Fare'].mean()}")
print(f"alive: {df[df['Survived'] == 1]['Fare'].mean()}")

# %% [md]
# Различается, пассажиры с более высокой ценой билета имели больший шанс выжить

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

old_age_start = 30
mo = df[df["Sex"] == "male"]
my = mo[(mo["Age"] >= 18) & (mo["Age"] < old_age_start)]
mo = mo[mo["Age"] >= old_age_start]
print(f"young: {len(my[my['Survived'] == 1]) / len(my) * 100}%")
print(f"old:   {len(mo[mo['Survived'] == 1]) / len(mo) * 100}%")

# %% [md]
# Гипотеза верна

# %% [md]
# ### 3. Визуализация

# %% [md]
# Используя библиотеки matplotlib/sns/plotly/pandas и набор данных
# представленный ранее визуализируйте

# %% [md]
# 1. Постройте гистограмму распределения возростов пассажиров

# %%
df["Age"].plot.hist()


# %% [md]
# 2. Постройте гистограммы распределения цен для пассажиров разных классов

# %%

df2 = pd.DataFrame(
    {
        "1 class": df[df["Pclass"] == 1]["Fare"],
        "2 class": df[df["Pclass"] == 2]["Fare"],
        "3 class": df[df["Pclass"] == 3]["Fare"],
    }
)

sns.histplot(df2.melt(), x="value", hue="variable", multiple="dodge",
             shrink=0.75, bins=20)


# %% [md]
# 3. Постройте зависимость цены от возраста пассажира

# %%


fig, ax = plt.subplots()
plt.plot(df["Age"], df["Fare"], "o", ms=3)
ax.xaxis.set_ticks(np.arange(0, df["Age"].max() + 1, 5))
plt.grid()
plt.show()


# %% [md]
# 4. Постройте box plot отображающий распределение цен на билеты
# в разных классах

# %%

df2.plot.box(title="Age box diagram", ylabel="fare")


# %% [md]
# 5. Визуализируйте распределение долей выживших среди мужчин,
# женщин и детей(до 16 лет)

# %%
vars = [
    df[(df["Sex"] == "male") & (df["Age"] > 16)],
    df[(df["Sex"] == "female") & (df["Age"] > 16)],
    df[(df["Age"] <= 16)]
]
labels = ["male", "female", "children"]

fig = plt.figure()
rect = (0.1, 0.1, 0.9, 0.9)

ax.grid(False)
ax.tick_params(axis='both', left=False, bottom=False,
               labelbottom=False, labelleft=True)
ax = fig.add_axes(rect, polar=True, frameon=False)
ax.grid(False, axis="y")
ax.set_theta_direction(1)
ax.set_theta_zero_location('N')
ax.set_xticks(1.5*np.pi*np.linspace(0, 1, 11),
              labels=[f"{int(i * 100)}%" for i in np.linspace(0, 1, 11)])


ax.set_rgrids(range(len(vars)),
              labels=[f"  {i}" for i in labels],
              angle=0,
              fontsize=14, fontweight='bold',
              color='black', verticalalignment='center')

#

for i, v in enumerate(vars):
    ax.barh(i, 1*1.5*np.pi, color="gray")
    t = len(v[v["Survived"] == 1])/len(v)
    pos = t*1.5*np.pi
    ax.barh(i, pos)
plt.show()


# %% [md]
# 6. Сравните плотности распределения возростов выживших и
# умерших пассажиров, сделайте выводы

# %%

df2 = pd.DataFrame({
    "alive": df[df["Survived"] == 1]["Age"],
    "dead": df[df["Survived"] == 0]["Age"]
})
df2.plot.kde(ind=np.linspace(0, np.ceil(df["Age"].max()), 100), grid=True)

# %% [md]
# Пассажиры старше 30 лет не показывают значительной зависимости между
# возрастом и процентом выживания. Отклонения плотностей - дети до 12 лет
# процент выживаемости детей значительно выше
# это также можно проследить в максимальных значения графика. Так как
# количество детей составляет большую долю выживших, количество молодых
# взрослых составляет
# пропорционально меньшую долю


# %% [md]
# 7. Постройте круговую диаграмму отображающую выживаемость мужчин и
# женщин в разных классах обслуживания(визуализация должна быть
# интуитивно понятная без объяснений, женщины и мужчины
# соответствующих классов объеденены визуально в одну группу
# и 2 подгруппы)*

# \* Задание на доп. балл

# %%

vars = [
    df[(df["Pclass"] == 1)],
    df[(df["Pclass"] == 2)],
    df[(df["Pclass"] == 3)],
]

fig = plt.figure()
rect = (0.1, 0.1, 0.9, 0.9)

ax.grid(False)
ax.tick_params(axis='both', left=False, bottom=False,
               labelbottom=False, labelleft=True)
ax = fig.add_axes(rect, polar=True, frameon=False)
ax.grid(False, axis="y")
ax.set_theta_direction(1)
ax.set_theta_zero_location('N')
ax.set_xticks(1.5*np.pi*np.linspace(0, 1, 11),
              labels=[f"{int(i * 100)}%" for i in np.linspace(0, 1, 11)])

ax.set_rgrids(range(9),
              labels=[f"  {i // 3 + 1} класс"
              if (i % 3) == 1 else "" for i in range(9)],
              angle=0,
              fontsize=14, fontweight='bold',
              color='black', verticalalignment='center')

for i, v in enumerate(vars):
    ax.barh(i*3, 1*1.5*np.pi, color="gray")
    ax.barh(i*3+1, 1*1.5*np.pi, color="gray")
    m = v[v["Sex"] == "male"]
    f = v[v["Sex"] == "female"]
    mp = len(m[m["Survived"] == 1])/len(m)
    fp = len(f[f["Survived"] == 1])/len(f)
    ax.barh(i*3, mp*1.5*np.pi, color="blue", label="male")
    ax.barh(i*3+1, fp*1.5*np.pi, color="cyan", label="female")
handles, labels = plt.gca().get_legend_handles_labels()
labels, ids = np.unique(labels, return_index=True)
handles = [handles[i] for i in ids]
plt.legend(handles, labels, loc="lower right")
plt.show()
