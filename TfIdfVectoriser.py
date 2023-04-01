import math
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from collections import Counter, defaultdict
import string


class TfIdfVectoriser:
    stopwords_set = set(stopwords.words('english'))
    vocab_counter = Counter()
    porter_stemmer = PorterStemmer()
    term_to_index = defaultdict(int)
    vocab_counter = dict()
    vocab_counter_reduced = dict()
    token_to_sample_count = dict()
    min_stem_occurrence = 3

    def __init__(self,
                 raw_dataset_file_path: str,
                 output_root_dir_path: str,
                 raw_data_first_n_rows: int,
                 limit_nrows: bool = False,
                 min_stem_occurrence: int = 3
                 ):
        self.raw_dataset_file_path = raw_dataset_file_path
        self.output_root_dir_path = output_root_dir_path
        self.raw_data_first_n_rows = raw_data_first_n_rows
        self.limit_nrows = limit_nrows
        self.min_stem_occurrence = min_stem_occurrence

    def process_sample_text(self, raw_text: str):
        # split
        tokens = word_tokenize(raw_text)
        # to lower case, strip new line feeds, carriage returns, and non-alpha
        tokens = [word.lower().replace('\r', '').replace('\n', '') for word in tokens if word.isalpha()]
        # remove punctuation
        table = str.maketrans('', '', string.punctuation)
        words_no_punct = [word.translate(table) for word in tokens]
        # remove stop words
        words_non_stopwords = [word[:20] for word in words_no_punct if word not in self.stopwords_set]
        # extract stems
        stems = [self.porter_stemmer.stem(word) for word in words_non_stopwords]
        self.vocab_counter.update(stems)
        return stems

    def filter_min_occurrence_stems(self, stems: list):
        return [stem for stem in stems if self.vocab_counter.get(stem) is not None]

    def count_num_of_samples_with_term(self, token_list: list):
        for token in token_list:
            if token in self.vocab_counter_reduced:
                self.token_to_sample_count[token] += 1

    def get_reduced_vocab(self):
        # vocab_counter_reduced is alphabetically ordered and will be used as to vectorise samples
        self.vocab_counter_reduced = {stem: count for stem, count in sorted(self.vocab_counter.items()) if
                                      count >= self.min_stem_occurrence}

    def fit(self):
        # covert to set the stopwords to remove from tokens

        if self.limit_nrows:
            df = pd.read_csv(self.raw_dataset_file_path, nrows=self.raw_data_first_n_rows)
        else:
            df = pd.read_csv(self.raw_dataset_file_path)

        # a sample is classified as 'profanity' if any of the other classes is non-zero
        df['profanity'] = np.where(df.iloc[:, 2:].sum(axis=1) > 0, 1, 0)
        # winsound.Beep(560, 1000)
        # extract tokens
        df['tokens'] = df.apply(lambda x: self.process_sample_text(x['comment_text']), axis=1)
        # filter out words that occur fewer than two times in the vocabulary
        # vocab_counter_reduced is alphabetically ordered and will be used as to vectorise samples
        vocab_counter_reduced = {stem: count for stem, count in sorted(self.vocab_counter.items()) if
                                 count >= self.min_stem_occurrence}
        df['tokens'] = df['tokens'].apply(lambda x: self.filter_min_occurrence_stems(x))

        # move the tokens column 2 (after the raw text)
        tokens_col = df.pop('tokens')
        df.insert(2, 'tokens', tokens_col)

        # calculate the num of samples each term appears in (needed for TF-IDF)
        term_to_sample_count = defaultdict(int)
        df['tokens'].apply(lambda x: self.count_num_of_samples_with_term(x))

        tf_idf_scores, tf_scores, idf_scores = vectorise_tf_idf(vocab_counter_reduced, term_to_sample_count,
                                                                df['profanity'], df['tokens'], profane_only=False)
        # profane_only_tf_idf_scores, profane_only_tf_scores, profane_only_idf_scores = vectorise_2(vocab_counter_reduced, term_to_sample_count, df['profanity'], df['tokens'], profane_only=True)
        np.savetxt(self.output_root_dir_path + 'tf_idf_scores_new.csv', tf_idf_scores, delimiter=',')
        # np.savetxt('C:\\Users\\kaspe\\OneDrive\\Pulpit\\test\\tf_idf_scores_old.csv', vectorised_matrix, delimiter=',')
        # save
        df.to_csv(processed_dataset_file_path)
        np.save(vectorised_matrix_file_path, tf_idf_scores)
        # pd.DataFrame(vectorised_matrix).to_excel(vectorised_matrix_file_path.split('.')[0] + '.xlsx', index=False)
        save_vocabulary(vocab_counter_reduced, vocabulary_file_path)
        # winsound.Beep(340, 3000)


    def vectorise_tf_idf(self,
                         term_to_sample_count: dict,
                         classification_column: pd.Series,
                         tokens_column: pd.Series,
                         profane_only: bool):
        num_samples = classification_column.size
        num_tokens = len(self.vocab_counter_reduced.items())

        tf_scores = np.zeros((num_samples, num_tokens))
        idf_scores = np.zeros((num_samples, num_tokens))

        for sample_index in range(0, num_samples):
            if profane_only:
                if classification_column.values[sample_index] == 0:
                    pass

            tf = defaultdict(int)

            for token in tokens_column.values[sample_index]:
                tf[token] += 1

            for term_index, term_count_tuple in enumerate(self.vocab_counter_reduced.items()):
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
