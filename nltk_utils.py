import nltk
import numpy as np

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()


def tokenize(text):
    return nltk.word_tokenize(text)


def stemming(text):
    return stemmer.stem(text.lower())


def bag_of_words(tokenized_text, all_words):
    """
    sentence = ["Hello", "How", "Are", "You"]
    words = ["hello", "hi", "i", "you", "bye"]
    bag = [ 1, 0, 0, 1, 0 ]
    :param tokenized_text:
    :param all_words:
    :return: bag of words
    """

    tokenized_text = [stemming(w) for w in tokenized_text]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for index, word in enumerate(all_words):
        if word in tokenized_text:
            bag[index] = 1.0

    return bag
