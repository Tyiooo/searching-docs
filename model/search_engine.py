import copy
import pandas as pd
import numpy as np
import sys
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import linear_kernel
from nltk.stem import PorterStemmer


class SearchEngine():
    replace_words = {'&': '_and_', 'unknown': ' '}

    def __init__(self, text_column='name', id_column='id'):
        self.text_column = text_column
        self.id_column = id_column
        pass

    def fit(self, df, ngram_range=(1, 3), perform_stem=True):
        self.df = df
        self.perform_stem = perform_stem
        doc_df = self.preprocess(df)
        stopWords = stopwords.words('english')
        self.vectoriser = CountVectorizer(stop_words=stopWords, ngram_range=ngram_range)
        train_vectorised = self.vectoriser.fit_transform(doc_df)
        self.transformer = TfidfTransformer()
        self.transformer.fit(train_vectorised)
        self.fitted_tfidf = self.transformer.transform(train_vectorised)

    def preprocess(self, df):
        result = df[self.text_column]
        result = np.core.defchararray.lower(result.values.astype(str))
        for word in self.replace_words:
            result = np.core.defchararray.replace(result, word, self.replace_words[word])
        if self.perform_stem:
            result = self.stem_array(result)
        return result

    def preprocess_query(self, query):
        result = query.lower()
        for word in self.replace_words:
            result = result.replace(word, self.replace_words[word])
        if self.perform_stem:
            result = self.stem_document(result)
        return result

    def stem_array(self, v):
        result = np.array([self.stem_document(document) for document in v])
        return result

    def stem_document(self, text):
        ps = PorterStemmer()
        result = [ps.stem(word) for word in text.split(" ")]
        result = ' '.join(result)
        return result

    def get_results(self, query, max_rows=10):
        score = self.get_score(query)
        results_df = copy.deepcopy(self.df)
        results_df['ranking_score'] = score
        results_df = results_df.loc[score > 0]
        results_df = results_df.iloc[np.argsort(-results_df['ranking_score'].values)]
        results_df = results_df.head(max_rows)
        return results_df

    def get_score(self, query):
        query_vectorised = self.vectoriser.transform([query])
        query_tfidf = self.transformer.transform(query_vectorised)
        cosine_similarities = linear_kernel(self.fitted_tfidf, query_tfidf).flatten()
        return cosine_similarities


