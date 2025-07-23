# 위의 DataFrame을 생성할 때 data 부분을 ndarray로 변경해보세요
import pandas as pd
import numpy as np

data = [[100, 120, 140, 115],
        [110, 130, 150, 97],
        [130, 160, 190, 125],
        [90, 110, 130, 110],
        [105, 125, 150, 105]]

arr = np.array(data)
df = pd.DataFrame(arr,
               columns=["1분기", "2분기","3분기","4분기"],
                index=["team1", "team2","team3","team4","team5"])
print(df)