"""
@Description: Converting categorical features to numerical-one-hot encoding and ordinal encoding
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-05-29 20:45:28
"""

from sklearn.feature_extraction import DictVectorizer

X_dict = [{'interest': 'tech', 'occupation': 'professional'},
          {'interest': 'fashion', 'occupation': 'student'},
          {'interest': 'fashion', 'occupation': 'professional'},
          {'interest': 'sports', 'occupation': 'student'},
          {'interest': 'tech', 'occupation': 'student'},
          {'interest': 'tech', 'occupation': 'retired'},
          {'interest': 'sports', 'occupation': 'professional'}]
dict_one_hot_encoder = DictVectorizer(sparse=False)
X_encoded = dict_one_hot_encoder.fit_transform(X_dict)

new_dict = [{'interest': 'sports', 'occupation': 'retired'}]
new_encoded = dict_one_hot_encoder.transform(new_dict)
# dict_one_hot_encoder.inverse_transform(new_encoded)

# DictVectorizer自动忽略之前未见过的categorical variables
new_dict = [{'interest': 'unknown_interest',
             'occupation': 'retired'},
            {'interest': 'tech', 'occupation':
             'unseen_occupation'}]
new_encoded = dict_one_hot_encoder.transform(new_dict)

import pandas as pd
df = pd.DataFrame({'score':['low', 'high', 'medium', 'medium', 'low']})
mapping = {'low': 1, 'medium': 2, 'high': 3}
df['score'] = df['score'].replace(mapping)