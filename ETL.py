import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from collections import Counter
import string


def process_sample_text(raw_text: str, stopwords_set: set, porter_stemmer: PorterStemmer, vocab_counter: Counter):
    # split
    tokens = word_tokenize(raw_text)
    # to lower case, strip new line feeds, carriage returns, and non-alpha
    tokens = [word.lower().replace('\r', '').replace('\n', '') for word in tokens if word.isalpha()]
    # remove punctuation
    table = str.maketrans('', '', string.punctuation)
    words_no_punct = [word.translate(table) for word in tokens]
    # remove stop words
    words_non_stopwords = [word[:20] for word in words_no_punct if word not in stopwords_set]
    # extract stems
    stems = [porter_stemmer.stem(word) for word in words_non_stopwords]
    vocab_counter.update(stems)
    return stems


def filter_min_occurrence_stems(stems: list, vocab_counter: dict):
    return [stem for stem in stems if vocab_counter.get(stem) is not None]


def count_num_of_samples_with_term(token_list: list, token_to_sample_count: dict, vocab_counter_reduced: dict):
    for token in token_list:
        if token in vocab_counter_reduced:
            token_to_sample_count[token] += 1


def save_vocabulary(vocabulary_counter: dict, vocabulary_file_path: str):
    with open(vocabulary_file_path, 'w', encoding='utf-8') as vocab_file:
        for stem, count in vocabulary_counter.items(): vocab_file.write(f'{stem},{count}\n')



def clean_dataset(df: pd.DataFrame):
    porter_stemmer = PorterStemmer()
    stopwords_set = set(stopwords.words('english'))
    # a sample is classified as 'profanity' if any of the other classes is non-zero
    df['profanity'] = np.where(df.iloc[:, 2:].sum(axis=1) > 0, 1, 0)

    # extract tokens
    df['clean'] = df.apply(lambda x: _process_sample_text(x['comment_text'], porter_stemmer, stopwords_set), axis=1)

    # move the tokens column 2 (after the raw text)
    tokens_col = df.pop('clean')
    df.insert(2, 'clean', tokens_col)
    return df


def _process_sample_text(raw_text: str, porter_stemmer: PorterStemmer, stopwords_set: set):
    # split
    tokens = word_tokenize(raw_text)
    # to lower case, strip new line feeds, carriage returns, and non-alpha
    tokens = [word.lower().replace('\r', '').replace('\n', '') for word in tokens if word.isalpha()]
    # remove punctuation
    table = str.maketrans('', '', string.punctuation)
    words_no_punct = [word.translate(table) for word in tokens]
    # remove stop words
    words_non_stopwords = [word[:20] for word in words_no_punct if word not in stopwords_set]
    # extract stems
    stems = [porter_stemmer.stem(word) for word in words_non_stopwords]
    return ' '.join(stems)

