import numpy as np
import pandas as pd

df.to_csv('/content/drive/MyDrive/Python_AI/data.csv')
print('csv 파일에 저장')

# index_col=0 은 인덱스로 사용되는 컬럼은 0번째이다.
df2 = pd.read_csv('/content/drive/MyDrive/Python_AI/data.csv', index_col=0)
# df2 = pd.read_csv('/content/drive/MyDrive/Python_AI/data.csv')


data = [
    [10, 40, 30, 40],
    [11, 25, 20, 50],
    [12, 24, 32, 70],
    [13, 25, 33, 40],
    [12, 21, 33, 41]
]

df = pd.DataFrame(data, columns=['1분기', '2분기', '3분기', '4분기'],  index=['team1', 'team2', 'team3', 'team4', 'team5'])


# TypeError: Sawon.add_sawon() missing 1 required positional argument: 'sPhone'
# Sawon.add_sawon() missing 1 required positional argument: 'info'
