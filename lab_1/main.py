import numpy as np
import matplotlib.pyplot as plt

# Данные
data1 = np.array([[3.1, 1.64],
                  [8, 3.64],
                  [12.4, 5.64],
                  [16.05, 7.64],
                  [20.1, 9.64],
                  [24.7, 11.64]])

data2 = np.array([[2.8, 2.23],
                  [4.4, 3.23],
                  [5.8, 4.23],
                  [7.1, 5.23],
                  [20.2, 7.23],
                  [21.6, 8.23],
                  [23, 9.23],
                  [24.3, 10.23],
                  [25.9, 11.23]])

# Разделение данных на X и φ
x1: np.ndarray = data1[:, 0]
q1: np.ndarray = data1[:, 1]
x2: np.ndarray = data2[:, 0]
q2: np.ndarray = data2[:, 1]


# Функция для вычисления коэффициентов линейной регрессии
def linear_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    n: int = len(x)
    k: float = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - (np.sum(x))**2)
    b: float = (np.sum(y) - k * np.sum(x)) / n
    return k, b


# Вычисление коэффициентов для обоих наборов данных
k1, b1 = linear_regression(x1, q1)
k2, b2 = linear_regression(x2, q2)

# Вычисление значений x для линии тренда
x_range1 = np.linspace(min(x1), max(x1), 100)
x_range2 = np.linspace(min(x2), max(x2), 100)

# Вычисление значений y для линии тренда
y_trend1 = k1 * x_range1 + b1
y_trend2 = k2 * x_range2 + b2



# Построение первого графика
plt.figure(figsize=(12, 6))
plt.scatter(x1, q1, color='#FF6500', label='Потенциал, В')
plt.plot(x_range1, y_trend1, color='#1E3E62', linestyle='--', label='Линия тренда')
plt.title('Без проводящего тела')
plt.xlabel('X')
plt.ylabel('φ')

# Подпись значений φ
for i in range(len(q1)):
    plt.text(x1[i], q1[i], f'{q1[i]:.2f}', fontsize=10, verticalalignment='bottom')

# Настройка сетки
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.xticks(np.arange(0, max(x1) + 2, 2))  # Настройка делений по оси X с шагом 2
plt.yticks(np.arange(0, max(q1) + 2, 2))  # Настройка делений по оси Y с шагом 2
plt.gca().set_aspect('equal', adjustable='box')  # Установить равные масштабы по осям
plt.legend()
plt.show()

# Построение второго графика
plt.figure(figsize=(12, 6))
plt.scatter(x2, q2, color='#FF6500', label='Потенциал, В')
plt.plot(x_range2, y_trend2, color='#1E3E62', linestyle='--', label='Линия тренда')
plt.title('С проводящим телом')
plt.xlabel('X')
plt.ylabel('φ')

# Подпись значений φ
for i in range(len(q2)):
    plt.text(x2[i], q2[i], f'{q2[i]:.2f}', fontsize=9, verticalalignment='bottom')

# Настройка сетки
plt.grid(which='both', linestyle='--', linewidth=0.5)
# Деление с шагом 2
plt.xticks(np.arange(0, max(x2) + 2, 2))
plt.yticks(np.arange(0, max(q2) + 2, 2)) 
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
