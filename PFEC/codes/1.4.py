# Replacing missing values with an arbitrary number

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from feature_engine.imputation import ArbitraryNumberImputer

data = pd.read_csv('../data/credit_approval_uci.csv')

X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis='columns'),
                                                    data['target'],
                                                    test_size=.3,
                                                    random_state=37)

arbitrary_cols = ['A2', 'A3', 'A8', 'A11']
print(X_train[arbitrary_cols].max().max())

X_train[arbitrary_cols] = X_train[arbitrary_cols].fillna(99)
X_test[arbitrary_cols] = X_test[arbitrary_cols].fillna(99)
print(X_train[arbitrary_cols].isnull().any().any())
print(X_test[arbitrary_cols].isnull().any().any())


# using scikit-learn
X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis='columns'),
                                                    data['target'],
                                                    test_size=.3,
                                                    random_state=37)
imputer = SimpleImputer(strategy='constant', fill_value=99)
imputer.fit(X_train[arbitrary_cols])
X_train[arbitrary_cols] = imputer.transform(X_train[arbitrary_cols])
X_test[arbitrary_cols] = imputer.transform(X_test[arbitrary_cols])
print(X_train[arbitrary_cols].isnull().any().any())
print(X_test[arbitrary_cols].isnull().any().any())


# using feature-engine
X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis='columns'),
                                                    data['target'],
                                                    test_size=.3,
                                                    random_state=37)
imputer = ArbitraryNumberImputer(arbitrary_number=99, variables=arbitrary_cols)

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
print(X_train[arbitrary_cols].isnull().any().any())
print(X_test[arbitrary_cols].isnull().any().any())