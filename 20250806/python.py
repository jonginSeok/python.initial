import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split


# 난수 고정
np.random.seed(11)

# X: 0~10 사이의 숫자 100개
X = np.linspace(0, 10, 100)
# Y: 2 * X + 약간의 noise
Y = 2 * X + 1 + np.random.normal(0, 1, size=X.shape)

# Pandas DataFrame으로 보기 좋게
df = pd.DataFrame({"X": X, "Y": Y})

# 데이터 시각화
sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.scatterplot(x="X", y="Y", data=df, color="blue")
plt.title(" X vs Y Scatter")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# 차원 추가: (100,) → (100,1)
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
Y_train = Y_train.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)

model = tf.keras.Sequential(
    [tf.keras.layers.Dense(1, input_shape=(1,))]  # 선형 회귀는 출력 1개
)

model.compile(optimizer="adam", loss="mse")

history = model.fit(X_train, Y_train, epochs=2000, verbose=1)

plt.plot(history.history["loss"])
plt.title(" Training Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# 테스트 데이터 예측
Y_pred = model.predict(X_test)

# 시각화
plt.figure(figsize=(8, 5))
plt.scatter(X_test, Y_test, label="Real", color="blue")
plt.scatter(X_test, Y_pred, label="Predicted", color="red")
plt.title("🔍 Real vs Prediction (Test Set)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

# 전체 X로 예측
X_full = X.reshape(-1, 1)
Y_full_pred = model.predict(X_full)

# 시각화
plt.figure(figsize=(8, 5))
plt.scatter(X, Y, label="Real", color="gray")
plt.plot(X, Y_full_pred, color="red", label="Prediction", linewidth=2)
plt.title("Regression Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
