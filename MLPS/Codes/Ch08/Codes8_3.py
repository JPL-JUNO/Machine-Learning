# """
# @Description: Working with bigger data – online algorithms and out-of-core learning
# @Author(s): Stephen CUI
# @LastEditor(s): Stephen CUI
# @CreatedTime: 2023-06-19 16:55:52
# """

# import sys
# sys.path.append('./')
# sys.path.append('../')
# from utilsML.funcs import tokenizer2
# import re
# import numpy as np


# def stream_docs(path: str) -> tuple:
#     """define a generator function, stream_docs, that reads in and returns one document at a time

#     Args:
#         path (str): file path

#     Returns:
#         tuple: tuple of text and label

#     Yields:
#         Iterator[tuple]: (text label)
#     """
#     with open(path, 'r', encoding='utf-8') as csv:
#         next(csv)  # skip header
#         for line in csv:
#             text, label = line[:-3], int(line[-2])
#             yield text, label


# def get_minibatch(doc_stream, size: int):
#     """take a document stream from the stream_docs function and return a particular number of documents specified by the size parameter

#     Args:
#         doc_stream (_type_): _description_
#         size (int): _description_

#     Returns:
#         _type_: _description_
#     """
#     docs, y = [], []
#     try:
#         for _ in range(size):
#             text, label = next(doc_stream)
#             docs.append(text)
#             y.append(label)
#     except StopIteration:
#         return None, None
#     return docs, y


from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
# vect = HashingVectorizer(decode_error='ignore',
#                          n_features=2**21, preprocessor=None, tokenizer=tokenizer2)
# clf = SGDClassifier(loss='log_loss', random_state=1)
# doc_stream = stream_docs(path='movie_data.csv')
import pyprind
# p_bar = pyprind.ProgBar(45)
# classes = np.array([0, 1])
# for _ in range(45):
#     X_train, y_train = get_minibatch(doc_stream, size=1000)
#     if not X_train:
#         break
#     X_train = vect.transform(X_train)
#     clf.partial_fit(X_train, y_train, classes=classes)
#     p_bar.update()

# X_test, y_test = get_minibatch(doc_stream, size=5_000)
# X_test = vect.transform(X_test)
# print(f'Accuracy: {clf.score(X_test, y_test):.3f}')
from nltk.corpus import stopwords
stop = stopwords.words('english')
import re
import numpy as np


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + \
        ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)  # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label


next(stream_docs(path='movie_data.csv'))


def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y


vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)


clf = SGDClassifier(loss='log', random_state=1)


doc_stream = stream_docs(path='movie_data.csv')


pbar = pyprind.ProgBar(45)

classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()


X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print(f'Accuracy: {clf.score(X_test, y_test):.3f}')


clf = clf.partial_fit(X_test, y_test)
