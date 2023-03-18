import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import string
import winsound


def process_text(raw_text, stopwords_set, porter_stemmer):
    # split
    tokens = word_tokenize(raw_text)
    # to lower case, strip new line feeds, carriage returns, and non-alpha
    tokens = [word.lower().replace('\r', '').replace('\n', '') for word in tokens if word.isalpha()]
    # remove punctuation
    table = str.maketrans('', '', string.punctuation)
    words_no_punct = [word.translate(table) for word in tokens]
    # remove stop words
    words_non_stopwords = [word[:100] for word in words_no_punct if word not in stopwords_set]
    # extract stems
    stems = [porter_stemmer.stem(word) for word in words_non_stopwords]
    # print(stems[:20])
    return stems


stopwords_set = set(stopwords.words('english'))
porter_stemmer = PorterStemmer()

print(stopwords.words('english'))

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

df = pd.read_csv('data/train.csv')

print(df.columns)
print(df.iloc[:, 2:])
df['profanity'] = np.where(df.iloc[:, 2:].sum(axis=1) > 0, 1, 0)
print(df.head(10))
winsound.Beep(560, 1000)
df['tokens'] = df.apply(lambda x: process_text(x['comment_text'], stopwords_set, porter_stemmer), axis=1)
tokens_col = df.pop('tokens')
df.insert(2, 'tokens', tokens_col)

winsound.Beep(440, 1000)

df.to_csv('data/clean.csv')

winsound.Beep(340, 5000)




