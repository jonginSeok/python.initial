# y = wx + b
# 15 = w * 2 + b
# 초기값
w = 0.0
b = 0.0

# 학습 데이터
x = 2
y = 15

# 학습률
lr = 0.01

# 에포크 수
epochs = 200

for epoch in range(epochs):
    # 순전파 (Forward)
    y_pred = w * x + b
    loss = 0.5 * (y_pred - y) ** 2   # MSE 손실 mean sequare error

    # 역전파 (Backward / Gradient)
    dL_dy_pred = y_pred - y          # dL/dy_pred
    dL_dw = dL_dy_pred * x           # Chain rule: dL/dw = (dy_pred/dw) * (dL/dy_pred)
    dL_db = dL_dy_pred * 1           # Chain rule: dL/db = (dy_pred/db) * (dL/dy_pred)

    # 파라미터 업데이트 (Gradient Descent)
    w -= lr * dL_dw
    b -= lr * dL_db

    # 10회마다 출력
    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:3d}: Loss = {loss:.4f}, w = {w:.4f}, b = {b:.4f}")