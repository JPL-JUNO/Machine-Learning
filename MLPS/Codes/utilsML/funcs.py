"""
@Description: Functions that are used to provide convenience
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-14 20:47:56
"""
from scipy.special import comb
import math
import re


def ensemble_error(n_classifier: int, error: float):
    assert error <= 1
    assert error >= 0
    k_start = int(math.ceil(n_classifier / 2))
    probs = [comb(n_classifier, k) * error**k * (1 - error) **
             (n_classifier - k) for k in range(k_start, n_classifier + 1)]
    return sum(probs)


def preprocessor(text: str) -> list:
    text = re.sub('<[^>]*>', '', text)  # remove all of th HTML markup
    emotions = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emotions).replace('-', ''))
    return text


def tokenizer(text: str) -> list:
    return text.split()


from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()


def tokenizer_porter(text: str) -> list:
    return [porter.stem(word) for word in text.split()]


from nltk.corpus import stopwords
stop = stopwords.words('english')


def tokenizer2(text: str) -> list:
    text = preprocessor(text)
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


if __name__ == '__main__':
    pass
