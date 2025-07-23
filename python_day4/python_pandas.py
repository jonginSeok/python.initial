# 1. 데이터프레임 기본 개념과 생성
import pandas as pd

# Series 생성
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(s)

# DataFrame 생성
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Score': [85, 90, 95]
}
df = pd.DataFrame(data)
print(df)
