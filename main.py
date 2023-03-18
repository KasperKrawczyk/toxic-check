import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from collections import Counter
import string
import winsound


def process_sample_text(raw_text: str, stopwords_set: set, porter_stemmer: PorterStemmer, vocab_counter: Counter):
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
    vocab_counter.update(stems)
    return stems

def filter_min_occurrence_stems(stems: list, min_occurrence: int, vocab_counter: Counter):
    return [stem for stem in stems if vocab_counter.get(stem) >= min_occurrence]

def save_vocabulary(vocabulary_counter: Counter, vocabulary_file_path: str):
    with open(vocabulary_file_path, 'w') as vocab_file:
        for stem, count in vocabulary_counter.items(): vocab_file.write(f'{stem},{count}\n')


def process_dataset(dataset_file_path: str):
    # covert to set the stopwords to remove from tokens
    stopwords_set = set(stopwords.words('english'))
    vocab_counter = Counter()
    porter_stemmer = PorterStemmer()

    df = pd.read_csv(dataset_file_path)

    # a sample is classified as 'profanity' if any of the other classes is non-zero
    df['profanity'] = np.where(df.iloc[:, 2:].sum(axis=1) > 0, 1, 0)
    winsound.Beep(560, 1000)
    # extract tokens
    df['tokens'] = df.apply(lambda x: process_sample_text(x['comment_text'], stopwords_set, porter_stemmer, vocab_counter), axis=1)
    # filter out words that occur fewer than two times in the vocabulary
    df['tokens'] = df.apply(lambda x: filter_min_occurrence_stems(x['tokens'], 3, vocab_counter))
    print(vocab_counter.most_common(50))
    print("vocabulary list length = {:d}".format(len(vocab_counter)))
    # move the tokens column 2 (after the raw text)
    tokens_col = df.pop('tokens')
    df.insert(2, 'tokens', tokens_col)

    winsound.Beep(440, 1000)
    # save
    df.to_csv('data/clean.csv')
    save_vocabulary(vocab_counter, 'data/vocabulary.txt')
    winsound.Beep(340, 5000)

def vectorise(clean_dataset_file_path: str):
    df = pd.read_csv(clean_dataset_file_path)
    vocabulary_set = set()


def __main__():





