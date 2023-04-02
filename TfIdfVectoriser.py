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
    vocab_counter_reduced = dict()
    term_to_sample_count = defaultdict(int)
    min_stem_occurrence = 3

    def __init__(self, min_stem_occurrence: int = 3):
        self.has_been_fitted = False
        self.num_samples = None
        self.min_stem_occurrence = min_stem_occurrence

    def _process_sample_text(self, raw_text: str, is_being_fit: bool):
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
        if is_being_fit:
            self.vocab_counter.update(stems)
        else:
            # allow only stems present in the fitted vocabulary
            stems = [stem for stem in stems if stem in self.vocab_counter_reduced]
        return stems

    def _filter_min_occurrence_stems(self, stems: list):
        return [stem for stem in stems if self.vocab_counter.get(stem) is not None]

    def _count_num_of_samples_with_term(self, token_list: list):
        for token in token_list:
            if token in self.vocab_counter_reduced:
                self.term_to_sample_count[token] += 1

    def _get_reduced_vocab(self):
        # vocab_counter_reduced is alphabetically ordered and will be used as to vectorise samples
        return {stem: count for stem, count in sorted(self.vocab_counter.items()) if
                count >= self.min_stem_occurrence}

    def _save_vocabulary(self, vocabulary_file_path: str):
        with open(vocabulary_file_path, 'w', encoding='utf-8') as vocab_file:
            for stem, count in self.vocab_counter_reduced.items(): vocab_file.write(f'{stem},{count}\n')

    def fit_transform(self, df: pd.DataFrame):
        if self.has_been_fitted:
            raise RuntimeError('Attempt to fit TfIdfVectoriser the second time. Terminating.')
        else:
            self.has_been_fitted = True

        self.num_samples = df.shape[0]
        # a sample is classified as 'profanity' if any of the other classes is non-zero
        df['profanity'] = np.where(df.iloc[:, 2:].sum(axis=1) > 0, 1, 0)
        # winsound.Beep(560, 1000)
        # extract tokens
        df['tokens'] = df.apply(lambda x: self._process_sample_text(x['comment_text'], is_being_fit=True), axis=1)
        # filter out words that occur fewer than two times in the vocabulary
        # vocab_counter_reduced is alphabetically ordered and will be used as to vectorise samples
        self.vocab_counter_reduced = self._get_reduced_vocab()
        df['tokens'] = df['tokens'].apply(lambda x: self._filter_min_occurrence_stems(x))

        # move the tokens column 2 (after the raw text)
        tokens_col = df.pop('tokens')
        df.insert(2, 'tokens', tokens_col)

        # calculate the num of samples each term appears in (needed for TF-IDF)
        df['tokens'].apply(lambda x: self._count_num_of_samples_with_term(x))

        return df['profanity'].to_numpy(), \
               self._vectorise_tf_idf(df['profanity'], df['tokens'], profane_only=False, is_being_fit=True)

        # save
        # np.savetxt(self.output_root_dir_path + 'tf_idf_scores_new.csv', tf_idf_scores, delimiter=',')
        # df.to_csv(self.output_root_dir_path + '')
        # np.save(self.output_root_dir_path + '', tf_idf_scores)
        # self.save_vocabulary(self.output_root_dir_path + '')

    def transform(self, df: pd.DataFrame):
        # a sample is classified as 'profanity' if any of the other classes is non-zero
        df['profanity'] = np.where(df.iloc[:, 2:].sum(axis=1) > 0, 1, 0)
        # extract tokens
        df['tokens'] = df.apply(lambda x: self._process_sample_text(x['comment_text'], is_being_fit=False), axis=1)
        # filter out words that occur fewer than two times in the vocabulary

        # move the tokens column 2 (after the raw text)
        tokens_col = df.pop('tokens')
        df.insert(2, 'tokens', tokens_col)

        # calculate the num of samples each term appears in (needed for TF-IDF)

        return df['profanity'].to_numpy(), \
               self._vectorise_tf_idf(df['profanity'], df['tokens'], profane_only=False, is_being_fit=False)

    def _vectorise_tf_idf(self,
                          classification_column: pd.Series,
                          tokens_column: pd.Series,
                          profane_only: bool,
                          is_being_fit: bool):
        if is_being_fit:
            num_corpus_samples = classification_column.size
        else:
            num_corpus_samples = self.num_samples

        num_samples = classification_column.size
        num_tokens = len(self.vocab_counter_reduced.items())

        tf_scores = np.zeros((num_samples, num_tokens))
        idf_scores = np.zeros((num_samples, num_tokens))

        for sample_index in range(0, num_samples):
            if profane_only and classification_column.values[sample_index] == 0:
                pass

            tf = defaultdict(int)

            for token in tokens_column.values[sample_index]:
                tf[token] += 1

            for term_index, term_count_tuple in enumerate(self.vocab_counter_reduced.items()):
                term = term_count_tuple[0]
                term_count = self.term_to_sample_count[term]
                tf_scores[sample_index][term_index] = math.log10(tf[term] + 1)
                idf_scores[sample_index][term_index] = 0 \
                    if term_count == 0 \
                    else math.log10((1 + num_corpus_samples) / (1 + self.term_to_sample_count[term]))

        tf_idf_scores = tf_scores * idf_scores
        norms = np.apply_along_axis(np.linalg.norm, axis=1, arr=tf_idf_scores)
        for sample_index, row in enumerate(tf_idf_scores):
            norm = norms[sample_index]
            for term_index, value in enumerate(row):
                row[term_index] = 0 if norm == 0 else value / norm

        # add classification column
        # tf_idf_scores = np.c_[classification_column.to_numpy(), tf_idf_scores]

        return tf_idf_scores

    def _vectorise_tf_idf_single_sample(self, tokens: list[string]):
        num_tokens = len(self.vocab_counter_reduced.items())

        tf_scores = np.zeros(num_tokens)
        idf_scores = np.zeros(num_tokens)

        tf = defaultdict(int)

        for token in tokens:
            tf[token] += 1

        for term_index, term_count_tuple in enumerate(self.vocab_counter_reduced.items()):
            term = term_count_tuple[0]
            term_count = self.term_to_sample_count[term]
            tf_scores[term_index] = math.log10(tf.get(term, 0) + 1)
            idf_scores[term_index] = 0 \
                if term_count == 0 \
                else math.log10((1 + self.num_samples) / (1 + self.term_to_sample_count[term]))

        tf_idf_scores = tf_scores * idf_scores
        norm = np.apply_along_axis(np.linalg.norm, axis=0, arr=tf_idf_scores).item()

        for term_index, value in enumerate(tf_idf_scores):
            tf_idf_scores[term_index] = 0 if norm == 0 else value / norm

        return tf_idf_scores

    def process_new(self, input_text: string):
        tokens = self._process_sample_text(input_text, False)

        tf_idf_vector = self._vectorise_tf_idf_single_sample(tokens)

        return tf_idf_vector
