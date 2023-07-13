"""
@Description: Topic modeling with latent Dirichlet allocation
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-23 13:17:07
"""

import pandas as pd
df = pd.read_csv('movie_data.csv', encoding='utf-8')
df = df.rename({'0': 'review', '1': 'sentiment'})

from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer(stop_words='english',
                        max_df=.1, max_features=5_000)
X = count.fit_transform(df['review'].values)

from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(
    n_components=10, random_state=123, learning_method='batch')
X_topics = lda.fit_transform(X)
assert lda.components_.shape == (10, 5_000)

n_top_words = 5
feature_names = count.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_, 1):
    print(f'Topic {topic_idx}')
    print(' '.join([feature_names[i]
          for i in topic.argsort()[:-n_top_words - 1:-1]]))

horror = X_topics[:, 5].argsort()[::-1]
for iter_idx, movie_idx in enumerate(horror[:3], 1):
    print(f'\nHorror movie # {iter_idx}"')
    print(df.loc[movie_idx, 'review'][:300], '...')
