import pandas as pd

# 2차원 리스트를 사용하여 데이터를 정의하고
# 컬럼명은 "1분기", "2분기","3분기","4분기" 으로 설정하고
# 인덱스는 "team1", "team2","team3","team4","team5"
# DataFrame 생성 및 표시
data = [[100, 120, 140, 115],
        [110, 130, 150, 97],
        [130, 160, 190, 125],
        [90, 110, 130, 110],
        [105, 125, 150, 105]]
df = pd.DataFrame(data,
                  columns=["1분기", "2분기","3분기","4분기"],
                  index=["team1", "team2","team3","team4","team5"])
print(df)

# 위의 DataFrame을 생성할 때 data 부분을 ndarray로 변경해보세요
import numpy as np

arr = np.array(data)
df = pd.DataFrame(arr,
               columns=["1분기", "2분기","3분기","4분기"],
                index=["team1", "team2","team3","team4","team5"])
print(df)

df.to_csv("data.csv")
print('csv 파일에 저장')

df2 = pd.read_csv("data.csv", index_col=0)  # 첫번째 컬럼이 인덱스임
print(df2)
print('csv 파일 읽음')

qt = df2['2분기']   # 한개의 컬럼 추출하기(Series)
print(type(qt))  # pandas.core.series.Series

arr = qt.values
print(arr)
print(type(arr))  # numpy.ndarray

row = df.loc['team3']
print(type(row))   # pandas.core.series.Series

row = df.iloc[2]
print(type(row))   # pandas.core.series.Series

df.loc['team6'] = [100,120,125,115]  # 행 추가
print(df)

df.iloc[6] = [110, 120,125,110]  # IndexError: iloc cannot enlarge its target object
print(df)


