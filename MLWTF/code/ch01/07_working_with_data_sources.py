"""
@Title: Working with data sources
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-09-06 17:21:45
@Description: 
"""

import tensorflow_datasets as tfds
import configparser
config = configparser.ConfigParser()
config.read("../data_link.ini")
# When you are importing a dataset for the first time, a bar will
# point out where you are as you download the dataset. If you
# prefer, you can deactivate it if you type the following:
# tfds.disable_progress_bar()

# 1. Iris data:
iris = tfds.load("iris", split="train")

# 2. Birth weight data:
import tensorflow as tf
birth_data_url = config.get("data_link", "Boston_housing_data")
path = tf.keras.utils.get_file(birth_data_url.split("/")[-1],
                               birth_data_url)


def map_line(x):
    """将由数字组成的字符串转为数字"""
    return tf.strings.to_number(tf.strings.split(x))


birth_file = tf.data.TextLineDataset(path).skip(1).map(map_line)


# 3. Boston housing data:
housing_url = config.get("data_link", "Boston_housing_data")
path = tf.keras.utils.get_file(housing_url.split("/")[-1],
                               housing_url)


def map_line(x: str):
    return tf.strings.to_number(tf.strings.split(x))


housing = tf.data.TextLineDataset(path).map(map_line)

# 4. MNIST handwriting data:
mnist = tfds.load("mnist", split=None)
mnist_train = mnist["train"]
mnist_test = mnist["test"]

# 5. Spam-ham text data.
zip_url = config.get("data_link", "Spam_ham_text_data")
path = tf.keras.utils.get_file(zip_url.split("/")[-1], zip_url, extract=True)

path = path.replace("smsspamcollection.zip", "SMSSpamCollection")


def split_text(x: str):
    return tf.strings.split(x, sep='\t')


text_data = tf.data.TextLineDataset(path).map(split_text)

# 6. Movie review data:

movie_data_url = config.get("data_link", "Movie_review_data")
path = tf.keras.utils.get_file(movie_data_url.split("/")[-1],
                               movie_data_url, extract=True)
path = path.replace(".tar.gz", '')
with open("movie_reviews.txt", 'w') as review_file:
    for response, filename in enumerate(["/rt-polarity.neg", "/rt-polarity.pos"]):
        with open(path + filename, 'r', encoding="utf-8", errors="ignore") as movie_file:
            for line in movie_file:
                review_file.write(str(response) + '\t' +
                                  line.encode('utf-8').decode())


def split_text(x):
    return tf.strings.split(x, sep='\t')


movies = tf.data.TextLineDataset("movie_reviews.txt").map(split_text)

# 7. CIFAR-10 image data:
ds, info = tfds.load("cifar10", shuffle_files=True, with_info=True)
print(info)
cifar_train = ds["train"]
cifar_test = ds["test"]

# 8. The works of Shakespeare text data:
shakespeare_url = config.get("data_link", "shakespeare")
path = tf.keras.utils.get_file(shakespeare_url.split("/")[-1], shakespeare_url)


def split_text(x):
    return tf.strings.split(x, sep="\n")


shakespeare_text = tf.data.TextLineDataset(path).map(split_text)

# 9. English-German sentence translation data:
import os
from zipfile import ZipFile
from urllib.request import Request, urlopen

sentence_url = config.get("data_link", "sentence_url")
r = Request(sentence_url, headers={
            "User-Agent": "Mozilla/5.0 (X11; U; Linux i686) Gecko/20071127 Firefox/2.0.0.11"})
b2 = [z for z in sentence_url.split('/') if ".zip" in z][0]
with open(b2, "wb") as target:
    target.write(urlopen(r).read())
with ZipFile(b2) as z:
    deu = [line.split('\t')[:2]
           for line in z.open("deu.txt").read().decode().split('\n')]
os.remove(b2)  # removes the zip file
with open("deu.txt", "wb") as deu_file:
    for line in deu:
        data = ",".join(line) + '\n'
        deu_file.write(data.encode('utf-8'))


def split_text(x):
    return tf.strings.split(x, sep=',')


text_data = tf.data.TextLineDataset("deu.txt").map(split_text)
