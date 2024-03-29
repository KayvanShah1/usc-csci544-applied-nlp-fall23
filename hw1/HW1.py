# Python Version: 3.10.12

import re
import unicodedata

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

import contractions

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB


class Config:
    RANDOM_STATE = 56
    # Place the downloaded Dataset at the same level as this python file
    DATA_PATH = "amazon_reviews_us_Office_Products_v1_00.tsv.gz"
    TEST_SPLIT = 0.2
    N_SAMPLES_EACH_CLASS = 50000
    NUM_TFIDF_FEATURES = 5000
    NUM_BOW_FEATURES = 5000


class DataLoader:
    @staticmethod
    def load_data(path):
        df = pd.read_csv(
            path,
            sep="\t",
            usecols=["review_headline", "review_body", "star_rating"],
            on_bad_lines="skip",
            memory_map=True,
        )
        return df


class DataProcessor:
    @staticmethod
    def filter_columns(df):
        return df.loc[:, ["review_body", "star_rating"]]

    @staticmethod
    def convert_star_rating(df):
        df["star_rating"] = pd.to_numeric(df["star_rating"], errors="coerce")
        df.dropna(subset=["star_rating"], inplace=True)
        return df

    @staticmethod
    def classify_sentiment(df):
        df["sentiment"] = df["star_rating"].apply(lambda x: 1 if x <= 3 else 2)
        return df

    @staticmethod
    def sample_data(df, n_samples, random_state):
        sampled_df = pd.concat(
            [
                df.query("sentiment==1").sample(n=n_samples, random_state=random_state),
                df.query("sentiment==2").sample(n=n_samples, random_state=random_state),
            ],
            ignore_index=True,
        ).sample(frac=1, random_state=random_state)

        sampled_df.drop(columns=["star_rating"], inplace=True)
        return sampled_df

    @staticmethod
    def basic_process_data(df):
        df = DataProcessor.filter_columns(df)
        df = DataProcessor.convert_star_rating(df)
        df = DataProcessor.classify_sentiment(df)

        df = DataProcessor.sample_data(df, Config.N_SAMPLES_EACH_CLASS, Config.RANDOM_STATE)

        # For consistency along the column, convert dtype the review_body to str
        df["review_body"] = df["review_body"].astype(str)
        return df


class TextCleaner:
    @staticmethod
    def unicode_to_ascii(s):
        return "".join(
            c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
        )

    @staticmethod
    def expand_contractions(text):
        return contractions.fix(text)

    @staticmethod
    def remove_email_addresses(text):
        return re.sub(r"[a-zA-Z0-9_\-\.]+@[a-zA-Z0-9_\-\.]+\.[a-zA-Z]{2,5}", " ", text)

    @staticmethod
    def remove_urls(text):
        return re.sub(r"\bhttps?:\/\/\S+|www\.\S+", " ", text)

    @staticmethod
    def remove_html_tags(text):
        return re.sub(r"<.*?>", "", text)

    @staticmethod
    def clean_text(text):
        text = TextCleaner.unicode_to_ascii(text.lower().strip())
        # replacing email addresses with empty string
        text = TextCleaner.remove_email_addresses(text)
        # replacing urls with empty string
        text = TextCleaner.remove_urls(text)
        # Remove HTML tags
        text = TextCleaner.remove_html_tags(text)
        # Expand contraction for eg., wouldn't => would not
        text = TextCleaner.expand_contractions(text)
        # creating a space between a word and the punctuation following it
        text = re.sub(r"([?.!,¿])", r" \1 ", text)
        text = re.sub(r'[" "]+', " ", text)
        # removes all non-alphabetical characters
        text = re.sub(r"[^a-zA-Z\s]+", "", text)
        # remove extra spaces
        text = re.sub(" +", " ", text)
        text = text.strip()
        return text


class TextPreprocessor:
    lemmatizer = WordNetLemmatizer()

    @staticmethod
    def get_stopwords_pattern():
        # Stopword list
        og_stopwords = set(stopwords.words("english"))

        # Define a list of negative words to remove
        neg_words = ["no", "not", "nor", "neither", "none", "never", "nobody", "nowhere"]
        custom_stopwords = [word for word in og_stopwords if word not in neg_words]
        pattern = re.compile(r"\b(" + r"|".join(custom_stopwords) + r")\b\s*")
        return pattern

    @staticmethod
    def pos_tagger(tag):
        if tag.startswith("J"):
            return wordnet.ADJ
        elif tag.startswith("V"):
            return wordnet.VERB
        elif tag.startswith("N"):
            return wordnet.NOUN
        elif tag.startswith("R"):
            return wordnet.ADV
        else:
            return None

    @staticmethod
    def lemmatize_text_using_pos_tags(text):
        words = nltk.pos_tag(word_tokenize(text))
        words = map(lambda x: (x[0], TextPreprocessor.pos_tagger(x[1])), words)
        lemmatized_words = [
            TextPreprocessor.lemmatizer.lemmatize(word, tag) if tag else word for word, tag in words
        ]
        return " ".join(lemmatized_words)

    @staticmethod
    def lemmatize_text(text):
        words = word_tokenize(text)
        lemmatized_words = [TextPreprocessor.lemmatizer.lemmatize(word) for word in words]
        return " ".join(lemmatized_words)

    pattern = get_stopwords_pattern()

    @staticmethod
    def preprocess_text(text):
        # replacing all the stopwords
        text = TextPreprocessor.pattern.sub("", text)
        text = TextPreprocessor.lemmatize_text(text)
        return text


clean_text_vect = np.vectorize(TextCleaner.clean_text)
preprocess_text_vect = np.vectorize(TextPreprocessor.preprocess_text)


def clean_and_process_data(path):
    # Place the downloaded Dataset at the same level as this python file
    df = DataLoader.load_data(path)

    # Do the basic processing of the dataset
    balanced_df = DataProcessor.basic_process_data(df)

    # Clean data
    avg_len_before_clean = balanced_df["review_body"].apply(len).mean()
    balanced_df["review_body"] = balanced_df["review_body"].apply(clean_text_vect)
    # Drop reviews that are empty
    balanced_df = balanced_df.loc[balanced_df["review_body"].str.strip() != ""]
    avg_len_after_clean = balanced_df["review_body"].apply(len).mean()

    # Preprocess data
    avg_len_before_preprocess = avg_len_after_clean
    balanced_df["review_body"] = balanced_df["review_body"].apply(preprocess_text_vect)
    avg_len_after_preprocess = balanced_df["review_body"].apply(len).mean()

    # Print Results
    print(f"{avg_len_before_clean:.2f}, {avg_len_after_clean:.2f}")
    print(f"{avg_len_before_preprocess:.2f}, {avg_len_after_preprocess:.2f}")

    return balanced_df


def evaluate_model(model, X_test, y_test):
    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    precision = precision_score(y_test, y_pred, average="binary")
    recall = recall_score(y_test, y_pred, average="binary")
    f1 = f1_score(y_test, y_pred, average="binary")

    return precision, recall, f1


def train_and_evaluate_model(model_class, X_train, y_train, X_test, y_test, **model_params):
    # Initialize model
    model = model_class(**model_params)

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate model
    precision, recall, f1 = evaluate_model(model, X_test, y_test)
    return model, precision, recall, f1


def main():
    balanced_df = clean_and_process_data(Config.DATA_PATH)

    # Splitting the reviews dataset
    X_train, X_test, y_train, y_test = train_test_split(
        balanced_df["review_body"],
        balanced_df["sentiment"],
        test_size=Config.TEST_SPLIT,
        random_state=Config.RANDOM_STATE,
    )

    # Feature Extraction
    tfidf_vectorizer = TfidfVectorizer(max_features=Config.NUM_TFIDF_FEATURES)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    count_vectorizer = CountVectorizer(max_features=Config.NUM_BOW_FEATURES)
    X_train_bow = count_vectorizer.fit_transform(X_train)
    X_test_bow = count_vectorizer.transform(X_test)

    # Train and evaluate Perceptron model using BoW features
    (
        _,
        precision_perceptron_bow,
        recall_perceptron_bow,
        f1_perceptron_bow,
    ) = train_and_evaluate_model(
        Perceptron, X_train_bow, y_train, X_test_bow, y_test, max_iter=4000
    )

    # Train and evaluate Perceptron model using TF-IDF features
    (
        _,
        precision_perceptron_tfidf,
        recall_perceptron_tfidf,
        f1_perceptron_tfidf,
    ) = train_and_evaluate_model(
        Perceptron, X_train_tfidf, y_train, X_test_tfidf, y_test, max_iter=4000
    )

    # Train and evaluate SVM model using BoW features
    _, precision_svm_bow, recall_svm_bow, f1_svm_bow = train_and_evaluate_model(
        LinearSVC, X_train_bow, y_train, X_test_bow, y_test, max_iter=2500
    )

    # Train and evaluate SVM model using TF-IDF features
    _, precision_svm_tfidf, recall_svm_tfidf, f1_svm_tfidf = train_and_evaluate_model(
        LinearSVC, X_train_tfidf, y_train, X_test_tfidf, y_test, max_iter=2500
    )

    # Train and evaluate Logistic Regression model using BoW features
    _, precision_lr_bow, recall_lr_bow, f1_lr_bow = train_and_evaluate_model(
        LogisticRegression, X_train_bow, y_train, X_test_bow, y_test, max_iter=4000
    )

    # Train and evaluate Logistic Regression model using TF-IDF features
    _, precision_lr_tfidf, recall_lr_tfidf, f1_lr_tfidf = train_and_evaluate_model(
        LogisticRegression, X_train_tfidf, y_train, X_test_tfidf, y_test, max_iter=4000
    )

    # Train and evaluate Naive Bayes model using BoW features
    _, precision_nb_bow, recall_nb_bow, f1_nb_bow = train_and_evaluate_model(
        MultinomialNB, X_train_bow, y_train, X_test_bow, y_test
    )

    # Train and evaluate Naive Bayes model using TF-IDF features
    _, precision_nb_tfidf, recall_nb_tfidf, f1_nb_tfidf = train_and_evaluate_model(
        MultinomialNB, X_train_tfidf, y_train, X_test_tfidf, y_test
    )

    # Print the results
    print(f"{precision_perceptron_bow:.4f} {recall_perceptron_bow:.4f} {f1_perceptron_bow:.4f}")
    print(
        f"{precision_perceptron_tfidf:.4f} {recall_perceptron_tfidf:.4f} {f1_perceptron_tfidf:.4f}"
    )

    print(f"{precision_svm_bow:.4f} {recall_svm_bow:.4f} {f1_svm_bow:.4f}")
    print(f"{precision_svm_tfidf:.4f} {recall_svm_tfidf:.4f} {f1_svm_tfidf:.4f}")

    print(f"{precision_lr_bow:.4f} {recall_lr_bow:.4f} {f1_lr_bow:.4f}")
    print(f"{precision_lr_tfidf:.4f} {recall_lr_tfidf:.4f} {f1_lr_tfidf:.4f}")

    print(f"{precision_nb_bow:.4f} {recall_nb_bow:.4f} {f1_nb_bow:.4f}")
    print(f"{precision_nb_tfidf:.4f} {recall_nb_tfidf:.4f} {f1_nb_tfidf:.4f}")


if __name__ == "__main__":
    main()
