import nltk
import numpy as np
nltk.download('punkt')
from  nltk.stem.porter import  PorterStemmer

# define some variables
stemmer = PorterStemmer()

def tokenization(sentence):
    return  nltk.word_tokenize(sentence)

def stemming(word):
    return  stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence,all_words):
    """
    :param tokenized_sentence: sentence after tokenization
    :param all_words: all unique words in dataset

    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]

    :return: array represents bag of words for the sentence
    """
    sentence_words = [stemming(word) for word in tokenized_sentence]  # stemming the words of the sentence as the words in all_words are stemmed

    bag = np.zeros(len(all_words), dtype = np.float32)
    for indx, word in enumerate(all_words):
        if word in sentence_words:
            bag[indx] = 1.0

    return bag

