"""
@Description: Preparing the IMDb movie review data for text processing
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-16 20:16:18
"""

import pyprind
import pandas as pd
import os
import sys
from packaging import version

base_path = 'aclImdb'
labels = {'pos': 1, 'neg': 0}
p_bar = pyprind.ProgBar(50_000, stream=sys.stdout)
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(base_path, s, l)
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path, file), 'r', encoding='utf-8') as in_file:
                txt = in_file.read()
            if version.parse(pd.__version__) >= version.parse('1.3.2'):
                x = pd.DataFrame([[txt, labels[l]]], columns=[
                                 'review', 'sentiment'])
                # df = pd.concat([df, x], ignore_index=False)
                df = pd.concat([df, x], ignore_index=True)
            else:
                df = df.append([[txt, labels[l]]], ignore_index=True)
            p_bar.update()
df.columns = ['review', 'sentiment']

# if version.parse(pd.__version__) >= version.parse('1.3.2'):
#     df.sample(frac=1, random_state=0).reset_index(drop=True)
import numpy as np
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('movie_data.csv', index=False, encoding='utf-8')

df = pd.read_csv('movie_data.csv', encoding='utf-8')
df = df.rename(columns={'0': 'review', '1': 'sentiment'})
df.sample(3)

assert df.shape == (50_000, 2)
