import math
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from collections import Counter, defaultdict
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


def count_num_of_samples_with_term(token_list: list, token_to_sample_count: dict, vocab_counter_reduced: dict):
    for token in token_list:
        if token in vocab_counter_reduced:
            token_to_sample_count[token] += 1


def save_vocabulary(vocabulary_counter: dict, vocabulary_file_path: str):
    with open(vocabulary_file_path, 'w', encoding='utf-8') as vocab_file:
        for stem, count in vocabulary_counter.items(): vocab_file.write(f'{stem},{count}\n')


def process_dataset(
        raw_dataset_file_path: str,
        processed_dataset_file_path: str,
        vectorised_matrix_file_path: str,
        vocabulary_file_path: str,
        min_stem_occurrence: int,
        raw_data_first_n_rows: int,
        limit_nrows: bool = False):
    # covert to set the stopwords to remove from tokens
    stopwords_set = set(stopwords.words('english'))
    vocab_counter = Counter()
    porter_stemmer = PorterStemmer()

    if limit_nrows:
        df = pd.read_csv(raw_dataset_file_path, nrows=raw_data_first_n_rows)
    else:
        df = pd.read_csv(raw_dataset_file_path)

    # a sample is classified as 'profanity' if any of the other classes is non-zero
    df['profanity'] = np.where(df.iloc[:, 2:].sum(axis=1) > 0, 1, 0)
    # winsound.Beep(560, 1000)
    # extract tokens
    df['tokens'] = df.apply(
        lambda x: process_sample_text(x['comment_text'], stopwords_set, porter_stemmer, vocab_counter), axis=1)
    # filter out words that occur fewer than two times in the vocabulary
    print(type(df['tokens']))
    # vocab_counter_reduced is alphabetically ordered and will be used as to vectorise samples
    vocab_counter_reduced = {stem: count for stem, count in sorted(vocab_counter.items()) if
                             count >= min_stem_occurrence}
    df['tokens'] = df['tokens'].apply(lambda x: filter_min_occurrence_stems(x, vocab_counter_reduced))
    # move the tokens column 2 (after the raw text)
    tokens_col = df.pop('tokens')
    df.insert(2, 'tokens', tokens_col)

    # calculate the num of samples each term appears in (needed for TF-IDF)
    term_to_sample_count = defaultdict(int)
    df['tokens'].apply(lambda x: count_num_of_samples_with_term(x, term_to_sample_count, vocab_counter_reduced))

    vectorised_matrix = create_vectorised_matrix(df, vocab_counter_reduced, term_to_sample_count)

    profane_samples_count = 0
    for row in vectorised_matrix:
        if row[0] == 1:
            profane_samples_count += 1
    print('Profane samples count: {}'.format(profane_samples_count))
    # winsound.Beep(440, 1000)

    tf_idf_scores, tf_scores, idf_scores = vectorise_2(vocab_counter_reduced, term_to_sample_count, df['profanity'], df['tokens'], profane_only=False)
    profane_only_tf_idf_scores, profane_only_tf_scores, profane_only_idf_scores = vectorise_2(vocab_counter_reduced, term_to_sample_count, df['profanity'], df['tokens'], profane_only=True)
    np.savetxt('C:\\Users\\kaspe\\OneDrive\\Pulpit\\test\\tf_idf_scores_new.csv', tf_idf_scores, delimiter=',')
    np.savetxt('C:\\Users\\kaspe\\OneDrive\\Pulpit\\test\\tf_idf_scores_old.csv', vectorised_matrix, delimiter=',')
    # save
    df.to_csv(processed_dataset_file_path)
    np.save(vectorised_matrix_file_path, vectorised_matrix)
    # pd.DataFrame(vectorised_matrix).to_excel(vectorised_matrix_file_path.split('.')[0] + '.xlsx', index=False)
    save_vocabulary(vocab_counter_reduced, vocabulary_file_path)
    # winsound.Beep(340, 3000)


def vectorise(sample_index: int, row: np.ndarray, vocab_counter_reduced: dict, term_to_sample_count: dict,
              classification_column: pd.Series,
              tokens_column: pd.Series):
    sample_tokens = tokens_column.values[sample_index]
    num_of_tokens_in_sample = len(sample_tokens)
    num_of_samples = classification_column.size

    sample_frequencies = defaultdict(int)
    # get TF (term frequency)
    for token in sample_tokens:
        sample_frequencies[token] += 1
    tf_scores = {token: math.log10(frequency + 1) for token, frequency in sample_frequencies.items()}

    # get IDF (inverse document frequency)
    idf_scores = {token: math.log10(num_of_samples / term_to_sample_count[token]) for token in sample_tokens}

    # get TF-IDF
    tf_idf_scores = {token: (tf_scores[token] * idf_scores[token]) for token in sample_tokens}

    # assign class
    row[0] = classification_column.values[sample_index]
    # create vector
    row_sum = 0
    for stem_index, stem in enumerate(vocab_counter_reduced.items()):
        score = tf_idf_scores.get(stem[0], 0)
        row[stem_index + 1] = score
        row_sum += math.pow(score, 2)
    length = math.sqrt(row_sum)
    # normalise
    if row_sum > 0:
        for i in range(1, len(row) - 1):
            row[i] = row[i] / length

def vectorise_2(vocab_counter_reduced: dict,
                term_to_sample_count: dict,
                classification_column: pd.Series,
                tokens_column: pd.Series,
                profane_only: bool):
    num_samples = classification_column.size
    num_tokens = len(vocab_counter_reduced.items())

    tf_scores = np.zeros((num_samples, num_tokens))
    idf_scores = np.zeros((num_samples, num_tokens))

    for sample_index in range(0, num_samples):
        if profane_only:
            if classification_column.values[sample_index] == 0:
                pass

        tf = defaultdict(int)

        for token in tokens_column.values[sample_index]:
            tf[token] += 1

        for term_index, term_count_tuple in enumerate(vocab_counter_reduced.items()):
            term = term_count_tuple[0]
            term_count = term_to_sample_count[term]
            tf_scores[sample_index][term_index] = math.log10(tf[term] + 1)
            idf_scores[sample_index][term_index] = 0 if term_count == 0 else math.log10(num_samples / term_to_sample_count[term])

    tf_idf_scores = tf_scores * idf_scores
    norms = np.apply_along_axis(np.linalg.norm, axis=1, arr=tf_idf_scores)
    for sample_index, row in enumerate(tf_idf_scores):
        norm = norms[sample_index]
        for term_index, value in enumerate(row):
            row[term_index] = 0 if norm == 0 else value / norm

    # add classification column
    tf_idf_scores = np.c_[classification_column.to_numpy(), tf_idf_scores]

    return tf_idf_scores, tf_scores, idf_scores





def create_vectorised_matrix(dataframe: pd.DataFrame, vocab_counter_reduced: dict, term_to_sample_count: dict):
    classification_column = dataframe['profanity']
    tokens_column = dataframe['tokens']
    matrix = np.zeros(shape=(dataframe.shape[0], len(vocab_counter_reduced.items()) + 1))
    for sample_index, row in enumerate(matrix):
        vectorise(sample_index, row, vocab_counter_reduced, term_to_sample_count, classification_column, tokens_column)

    return matrix


if __name__ == '__main__':
    root_path = 'C:\\Users\\kaspe\\OneDrive\\Pulpit\\test\\'
    process_dataset(
        root_path + 'train_test.csv',
        root_path + 'clean_full.csv',
        root_path + 'vectorised_matrix.npy',
        root_path + 'vocabulary_full.csv',
        3,
        0)
