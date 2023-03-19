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
    words_non_stopwords = [word[:20] for word in words_no_punct if word not in stopwords_set]
    # extract stems
    stems = [porter_stemmer.stem(word) for word in words_non_stopwords]
    vocab_counter.update(stems)
    return stems


def filter_min_occurrence_stems(stems: list, vocab_counter: dict):
    return [stem for stem in stems if vocab_counter.get(stem) is not None]


def save_vocabulary(vocabulary_counter: dict, vocabulary_file_path: str):
    with open(vocabulary_file_path, 'w', encoding='utf-8') as vocab_file:
        for stem, count in vocabulary_counter.items(): vocab_file.write(f'{stem},{count}\n')


def process_dataset(raw_dataset_file_path: str, processed_dataset_file_path: str, vocabulary_file_path: str, min_stem_occurrence: int):
    # covert to set the stopwords to remove from tokens
    stopwords_set = set(stopwords.words('english'))
    vocab_counter = Counter()
    porter_stemmer = PorterStemmer()

    df = pd.read_csv(raw_dataset_file_path)

    # a sample is classified as 'profanity' if any of the other classes is non-zero
    df['profanity'] = np.where(df.iloc[:, 2:].sum(axis=1) > 0, 1, 0)
    winsound.Beep(560, 1000)
    # extract tokens
    df['tokens'] = df.apply(
        lambda x: process_sample_text(x['comment_text'], stopwords_set, porter_stemmer, vocab_counter), axis=1)
    # filter out words that occur fewer than two times in the vocabulary
    print(type(df['tokens']))
    # vocab_counter_reduced is alphabetically ordered and will be used as to vectorise samples
    vocab_counter_reduced = {stem: stem for stem, count in sorted(vocab_counter.items()) if count >= min_stem_occurrence}
    df['tokens'] = df['tokens'].apply(lambda x: filter_min_occurrence_stems(x, vocab_counter_reduced))
    # move the tokens column 2 (after the raw text)
    tokens_col = df.pop('tokens')
    df.insert(2, 'tokens', tokens_col)
    df['vectors'] = df['tokens'].apply(lambda x: vectorise(x, vocab_counter_reduced))

    winsound.Beep(440, 1000)
    # save
    df.to_csv(processed_dataset_file_path)
    save_vocabulary(vocab_counter_reduced, vocabulary_file_path)
    winsound.Beep(340, 5000)


def vectorise(stems: list, vocab_counter_reduced: dict):
    vector = np.array()
    stems_set_per_sample = set(stems)
    for stem, count in vocab_counter_reduced.items():
        vector.append(1 if stem in stems_set_per_sample else 0)
    return vector

def create_vectorised_matrix(dataframe: pd.DataFrame, vocab_counter_reduced: dict):
    matrix = np.zeros(shape=(dataframe.shape[0], len(vocab_counter_reduced.items()) + 1))
    for index, val in np.ndenumerate(matrix):
        if (index[1] == 0):
            matrix[index[0], index[1]] = dataframe['profanity'][index[0]]
        else:

    return matrix

if __name__ == '__main__':
    process_dataset('test/train_test.csv', 'test/clean_test.scv', 'test/vocabulary_test.txt', 3)
