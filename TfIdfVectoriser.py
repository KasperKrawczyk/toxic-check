import math
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from collections import Counter, defaultdict
import string


def _get_n_grams(tokens: list, n: int = 6):
    return [tokens[i: i + n] for i in range(len(tokens) - n + 1)]


class TfIdfVectoriser:
    stopwords_set = set(stopwords.words('english'))
    vocab_counter = Counter()
    porter_stemmer = PorterStemmer()
    term_to_index = defaultdict(int)
    vocab_counter_reduced = dict()
    df = defaultdict(int)
    min_stem_occurrence = 3

    def __init__(self, min_stem_occurrence: int = 3, min_idf_score: float = 3.2, n_gram_length: int = 1):
        self.has_been_fitted = False
        self.num_samples = None
        self.min_stem_occurrence = min_stem_occurrence
        self.min_idf_score = min_idf_score
        self.n_gram_length = n_gram_length

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
        if self.n_gram_length > 1:
            n_grams_list = _get_n_grams(stems)
            stems = [' '.join(n_gram) for n_gram in n_grams_list]
        if is_being_fit:
            self.vocab_counter.update(stems)
        else:
            # allow only stems present in the fitted vocabulary
            stems = [stem for stem in stems if stem in self.vocab_counter_reduced]
        return stems

    def _filter_min_occurrence_stems(self, stems: list):
        return [stem for stem in stems if stem in self.vocab_counter_reduced]

    def _get_df(self, token_list: list):
        for token in token_list:
            if token in self.vocab_counter_reduced:
                self.df[token] += 1

    def _get_reduced_vocab(self):
        if self.min_stem_occurrence > 1:
            return {stem: count for stem, count in sorted(self.vocab_counter.items()) if count >= self.min_stem_occurrence}
        else:
            return {stem: count for stem, count in sorted(self.vocab_counter.items())}

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
        # df = df[df['profanity'] == 1].copy()

        # extract tokens
        df['tokens'] = df.apply(lambda x: self._process_sample_text(x['comment_text'], is_being_fit=True), axis=1)
        # filter out words that occur fewer than two times in the vocabulary
        # vocab_counter_reduced is alphabetically ordered and will be used as to vectorise samples
        self.vocab_counter_reduced = self._get_reduced_vocab()
        if self.min_stem_occurrence > 1:
            df['tokens'] = df['tokens'].apply(lambda x: self._filter_min_occurrence_stems(x))

        # move the tokens column 2 (after the raw text)
        tokens_col = df.pop('tokens')
        df.insert(2, 'tokens', tokens_col)

        # calculate the num of samples each term appears in (needed for TF-IDF)
        df['tokens'].apply(lambda x: self._get_df(x))

        return df['profanity'].to_numpy(), self._vectorise_tf_idf(df['profanity'], df['tokens'])

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

        return df['profanity'].to_numpy(), self._vectorise_tf_idf(df['profanity'], df['tokens'])

    def _vectorise_tf_idf(self,
                          classification_column: pd.Series,
                          tokens_column: pd.Series):

        num_sample_in_dataset = classification_column.size
        num_tokens_in_corpus = len(self.vocab_counter_reduced.items())

        idf_scores = self._get_idf_scores()
        tf_scores = np.zeros((num_sample_in_dataset, num_tokens_in_corpus))

        # calculate TF scores
        for sample_index in range(0, num_sample_in_dataset):

            sample_tokens = tokens_column.values[sample_index]
            tf = defaultdict(int)

            for token in sample_tokens:
                tf[token] += 1

            for term_index, term_count_tuple in enumerate(self.vocab_counter_reduced.items()):
                term = term_count_tuple[0]
                term_freq = tf[term]
                if term_freq > 0:
                    tf_scores[sample_index][term_index] = tf[term] / len(tokens_column.values[sample_index])
                else:
                    tf_scores[sample_index][term_index] = 0

        tf_idf_scores = tf_scores * idf_scores
        norms = np.apply_along_axis(np.linalg.norm, axis=1, arr=tf_idf_scores)
        for sample_index, row in enumerate(tf_idf_scores):
            norm = norms[sample_index]
            for term_index, value in enumerate(row):
                row[term_index] = 0 if norm == 0 else value / norm

        # add classification column
        # tf_idf_scores = np.c_[classification_column.to_numpy(), tf_idf_scores]

        return tf_idf_scores

    def _get_idf_scores(self):
        num_tokens = len(self.vocab_counter_reduced.items())
        idf_scores = np.zeros(num_tokens)
        num_corpus_samples = self.num_samples + 1

        # calculate IDF scores
        # both num_corpus_samples and term_count get incremented by one to apply idf smoothing
        # log gets incremented by one so that terms with idf == 0 have at least minimal weight to them
        for term_index, term_count_tuple in enumerate(self.vocab_counter_reduced.items()):
            term = term_count_tuple[0]
            term_count = self.df[term]
            idf_score = (1 + math.log10((1 + num_corpus_samples) / (1 + term_count)))
            idf_scores[term_index] = idf_score

        return idf_scores

    def _vectorise_tf_idf_single_sample(self, tokens: list[string]):
        num_tokens = len(self.vocab_counter_reduced.items())

        tf_scores = np.zeros(num_tokens)
        idf_scores = np.zeros(num_tokens)

        tf = defaultdict(int)

        for token in tokens:
            tf[token] += 1

        for term_index, term_count_tuple in enumerate(self.vocab_counter_reduced.items()):
            term = term_count_tuple[0]
            term_count = self.df[term]
            tf_scores[term_index] = math.log10(tf.get(term, 0) + 1)
            idf_scores[term_index] = 0 \
                if term_count == 0 \
                else math.log10((1 + self.num_samples) / (1 + self.df[term]))

        tf_idf_scores = tf_scores * idf_scores
        norm = np.apply_along_axis(np.linalg.norm, axis=0, arr=tf_idf_scores).item()

        for term_index, value in enumerate(tf_idf_scores):
            tf_idf_scores[term_index] = 0 if norm == 0 else value / norm

        return tf_idf_scores

    def process_new(self, input_text: string):
        tokens = self._process_sample_text(input_text, False)

        tf_idf_vector = self._vectorise_tf_idf_single_sample(tokens)

        return tf_idf_vector
