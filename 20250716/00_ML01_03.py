import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

# y = 2x^2 + 3x + 1
# 2차 방정식과 가중치, 편향값 시각화

# 파라미터 설정
w1 = 2
w2 = 3
b = 1

# x 값 100개 생성 (예: -5부터 5까지 균등 분포)
x = np.linspace(-5, 5, 100)

# y 값 계산
y = w1 * x**2 + w2 * x + b

print("x=", x)
print("y=", y)

# 시각화
plt.figure(figsize=(5, 3))
plt.plot(x, y, label=f"y = {w1} * x^2 + {w2} * x + {b}", color="blue")
plt.title(f"y = {w1} * x^2 + {w2} * x + {b}")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")  # x축과 y축의 비율을 동일하게 설정
plt.grid(True)
plt.legend()
plt.show()
