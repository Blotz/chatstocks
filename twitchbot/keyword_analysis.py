from sklearn.feature_extraction.text import CountVectorizer
import joblib
import pandas as pd
import numpy as np
import scipy.sparse
import os
import pathlib


class KeywordAnalysis:
    VECTORIZER_PATH: pathlib.Path = pathlib.Path("vectorizer.pkl")
    MATRIX_PATH: pathlib.Path = pathlib.Path("document_term_matrix.npz")

    def __init__(self) -> None:
        # load matrix + vectorizer
        self.vectorizer: CountVectorizer = self.load_vectorizer(self.VECTORIZER_PATH)
        self.document_term_matrix = self.load_matrix(self.MATRIX_PATH)

    def init_vectorizer(self) -> CountVectorizer:
        return CountVectorizer(stop_words="english")

    def load_vectorizer(self, path: pathlib.Path) -> CountVectorizer:
        if os.path.exists(path):
            return joblib.load(path)
        else:
            return self.init_vectorizer()

    def load_matrix(self, path: pathlib.Path) -> scipy.sparse.csr_matrix:
        if os.path.exists(path):
            return scipy.sparse.load_npz(path)
        else:
            return self.vectorizer.fit_transform(["test123"])

    def save_vectorizer(self, path) -> None:
        joblib.dump(self.vectorizer, path)

    def save_matrix(self, path) -> None:
        scipy.sparse.save_npz(path, self.matrix)

    def fit(self, data: pd.Series) -> None:
        # Fit new data
        new_vectorizer = self.init_vectorizer()
        new_vectorizer.fit(data)

        # Get features
        old_features = self.vectorizer.get_feature_names_out()
        new_features = new_vectorizer.get_feature_names_out()

        missing_features = np.setdiff1d(new_features, old_features)

        if len(missing_features) > 0:
            # Add the missing features to the old features
            old_features = np.append(old_features, missing_features)
            old_features = np.sort(old_features)

            # Update the vocabulary
            vocab_dict = {feature: i for i, feature in enumerate(old_features)}
            self.vectorizer.vocabulary_ = vocab_dict

            # Add what new features are missing and add them to the old features
            sorted_indices = np.argsort(old_features)

            # Resize the matrix to fit the vocab
            self.document_term_matrix.resize(
                (self.document_term_matrix.shape[0], len(old_features))
            )
            self.document_term_matrix = self.document_term_matrix[:, sorted_indices]

        # Generate the new matrix
        new_matrix = self.vectorizer.transform(data)

        # Add the new matrix to the old matrix
        self.document_term_matrix = scipy.sparse.vstack(
            [self.document_term_matrix, new_matrix]
        )

    def get_n_top_keywords(self, msg: str, n: int) -> list:
        tfidf_matrix = self.vectorizer.transform([msg])
        feature_names = self.vectorizer.get_feature_names_out()

        tfidf_scores = tfidf_matrix.toarray()[0]
        top_indices = tfidf_scores.argsort()[-n:][::-1]

        top_keywords = [(feature_names[i], tfidf_scores[i]) for i in top_indices]

        return top_keywords

    def get_n_associated_keywords(self, word: str, n: int) -> list:
        features = self.vectorizer.get_feature_names_out()
        if word not in features:
            return []

        co_occurrence_matrix = self.document_term_matrix.T * self.document_term_matrix
        co_occurrence_matrix.setdiag(0)

        word_index = np.where(features == word)[0][0]
        co_occurrences = co_occurrence_matrix[word_index].toarray()[0]
        top_indices = co_occurrences.argsort()[-n:][::-1]

        return [(features[i], co_occurrences[i]) for i in top_indices]

    def find_top_associated_keywords(self, threshold: int = 100) -> list:
        co_occurrence_matrix = self.document_term_matrix.T * self.document_term_matrix
        co_occurrence_matrix.setdiag(0)
        co_occurrence_matrix.setdiag(0)  # i dont know why it needs to be set twice

        features = self.vectorizer.get_feature_names_out()
        top_indices = np.where(co_occurrence_matrix.toarray() > threshold)

        top_associations = [
            (features[i], features[j], co_occurrence_matrix[i, j])
            for i, j in zip(*top_indices)
        ]

        return top_associations
