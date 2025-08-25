import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

"""
2025.07.16
"""

# y = w * x + b
# 직선의 방정식과 기울기(계수, 가중치, weight), 절편(편견,편향, bias)

# 파라미터 설정
w = 0.0000001
b = 3

# x 값 100개 생성 (예: -10부터 10까지 균등 분포)
x = np.linspace(-10, 10, 100)

# y 값 계산
y = w * x + b

print("x=", x)
print("y=", y)

# 시각화
plt.figure(figsize=(5, 3))
plt.plot(x, y, label=f"y = {w}x + {b}", color="blue")
plt.title(f"Linear Function: y = {w}x + {b}")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")  # x축과 y축의 비율을 동일하게 설정
plt.grid(True)
plt.legend()
plt.show()
