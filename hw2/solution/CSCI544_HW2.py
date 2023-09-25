import json
import os
import warnings

warnings.filterwarnings("ignore")

from collections import Counter
from typing import List

import numpy as np
import pandas as pd


class PathConfig:
    HW2_DIR = os.path.dirname(os.getcwd())
    OUTPUT_DIR = os.path.join(HW2_DIR, "solution", "output")

    DATA_PATH = os.path.join(HW2_DIR, "CSCI544_HW2", "data")
    VERIFICATION_DATA_PATH = os.path.join(HW2_DIR, "CSCI544_HW2", "verification")

    VOCAB_FILE_PATH = os.path.join(OUTPUT_DIR, "vocab.txt")
    HMM_MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "hmm.json")


class VocabConfig:
    UNKNOWN_TOKEN = "<unk>"
    THRESHOLD = 3
    FILE_HEADER = ["word", "index", "frequency"]

    VOCAB_FILE = PathConfig.VOCAB_FILE_PATH


class HMMConfig:
    HMM_MODEL_SAVED = PathConfig.HMM_MODEL_SAVE_PATH


class VocabularyGenerator:
    def __init__(
        self, threshold: int, unknown_token: str = None, save: bool = False, path: str = None
    ):
        """Initialize a VocabularyGenerator

        Args:
            threshold (int): Frequency threshold for rare words.
            unknown_token (str, optional): Token to replace rare words. Defaults to None.
            save (bool, optional): Flag to save the vocabulary. Default is True.
            path (str, optional): Path to save the vocabulary. Defaults to None.

        Usage:
            vocab_generator = VocabularyGenerator(threshold=3, unknown_token="<unk>")
            vocab_df = vocab_generator.generate_vocabulary(data, "sentence")
        """
        self.threshold = threshold
        self.unknown_token = (
            unknown_token if unknown_token is not None else VocabConfig.UNKNOWN_TOKEN
        )
        self._save = save

        if self._save and path is None:
            self.path = VocabConfig.VOCAB_FILE
        else:
            self.path = path

    def _count_word_frequency(self, data, sentence_col_name):
        word_freq = (
            data[sentence_col_name]
            .explode()
            .value_counts()
            .rename_axis("word")
            .reset_index(name="frequency")
        )
        return word_freq

    def generate_vocabulary(self, data: pd.DataFrame, sentence_col_name: str):
        """Generate a vocabulary from the provided dataset.

        Args:
            data (pd.DataFrame): The DataFrame containing the dataset.
            sentence_col_name (str): The name of the column containing sentences.

        Returns:
            pd.DataFrame: A DataFrame with the generated vocabulary.

        This method takes a DataFrame with sentences and generates a vocabulary based on word
        frequencies.
        It replaces words with frequencies less than the specified threshold with the unknown token
        ("<unk>").
        The resulting DataFrame is sorted by frequency and indexed.

        If the 'save' flag is set, the vocabulary will be saved to the specified path.

        Usage:
            ```py
            vocab_generator = VocabularyGenerator(threshold=3, unknown_token="<unk>")
            vocab_df = vocab_generator.generate_vocabulary(data, sentence_col_name)
            ```
        """
        word_freq_df = self._count_word_frequency(data, sentence_col_name)

        # Create a DataFrame
        # word_freq_df = pd.DataFrame(word_freq_list, columns=["word", "frequency"])

        # Replace words with frequency less than threshold with '<unk>'
        word_freq_df["word"] = word_freq_df.apply(
            lambda row: self.unknown_token if row["frequency"] <= self.threshold else row["word"],
            axis=1,
        )

        # # Group by 'Word' and aggregate by sum
        word_freq_df = word_freq_df.groupby("word", as_index=False)["frequency"].agg("sum")

        # Sort the DataFrame by frequency
        word_freq_df = word_freq_df.sort_values(by="frequency", ascending=False, ignore_index=True)

        # Add an index column
        word_freq_df["index"] = range(1, len(word_freq_df) + 1)

        if self._save:
            self.save_vocab(word_freq_df, self.path)

        return word_freq_df

    def save_vocab(self, word_freq_df, path):
        """Write your vocabulary to the file"""
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        with open(path, "w") as file:
            vocabulary = word_freq_df.to_records(index=False)
            for word, frequency, index in vocabulary:
                file.write(f"{word}\t{index}\t{frequency}\n")


class HMM:
    def __init__(self, vocab_file: str, labels: List[str]):
        """_summary_

        Args:
            train_data (pd.DataFrame): _description_
            vocab_file (str): _description_
        """
        self.vocab = self._read_vocab(vocab_file)
        self.labels = labels

        # Hidden Markov Model Parameters
        self.states = list()
        self.priors = None
        self.transitions = None
        self.emissions = None

    def _read_vocab(self, vocab_file: str):
        return pd.read_csv(vocab_file, sep="\t", names=VocabConfig.FILE_HEADER)

    def _initialize_params(self):
        self.states = list(self.labels)

        # N = Number of states i.e. number of distinct tags
        num_states = len(self.labels)
        # M = Number of observable symbols i.e. number of distinct words
        num_observations = len(self.vocab)

        # State transition probability matrix of size N * N
        self.transitions = np.zeros((num_states, num_states))

        # Obseravtion Emission probability matrix of size N * M
        self.emissions = np.zeros((num_states, num_observations))

        # Prior probability matrix of size N * 1
        self.prior = np.ones(num_states)

    def _compute_prior_params(self, train_data):
        num_sentences = len(train_data)

        state_occurrence = Counter()

        for sentence in train_data:
            # Ensure the sentence is not empty
            if sentence:
                # Get the label of the first word in the sentence
                label = sentence[0][1]
                state_occurrence[label] += 1

        self.priors = np.array([state_occurrence[state] / num_sentences for state in self.labels])

    def _compute_transition_params(self, train_data):
        labels_list = [label for sentence in train_data for _, label in sentence]
        label_indices = [self.states.index(label) for label in labels_list]

        for i in range(len(label_indices) - 1):
            curr_state = label_indices[i]
            next_state = label_indices[i + 1]
            self.transitions[curr_state, next_state] += 1

        # Handle cases where the probabilities is 0
        self.transitions = np.where(self.transitions == 0, 1e-10, self.transitions)

        row_agg = self.transitions.sum(axis=1)
        self.transitions = self.transitions / row_agg[:, np.newaxis]

    def _compute_emission_params(self, train_data):
        word_to_index = dict(zip(self.vocab["word"], self.vocab["index"]))

        for sentence in train_data:
            for word, label in sentence:
                state_idx = self.states.index(label)
                word_idx = word_to_index.get(word, word_to_index[VocabConfig.UNKNOWN_TOKEN]) - 1
                self.emissions[state_idx, word_idx] += 1

        # Handle cases where the probabilities is 0
        self.emissions = np.where(self.emissions == 0, 1e-10, self.emissions)

        row_agg = self.emissions.sum(axis=1)
        self.emissions = self.emissions / row_agg[:, np.newaxis]

    def fit(self, train_data: pd.DataFrame):
        self._initialize_params()
        self._compute_prior_params(train_data)
        self._compute_transition_params(train_data)
        self._compute_emission_params(train_data)

    @property
    def get_all_probability_matrices(self):
        return self.priors, self.transitions, self.emissions

    def save_model(self, file_path=None):
        if file_path is None:
            file_path = HMMConfig.HMM_MODEL_SAVED

        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        transition_prob = {
            f"({s1}, {s2})": p
            for s1 in self.states
            for s2 in self.states
            for p in [self.transitions[self.states.index(s1), self.states.index(s2)]]
        }

        emission_prob = {
            f"({s}, {w})": p
            for s in self.states
            for w, p in zip(self.vocab["word"], self.emissions[self.states.index(s), :])
        }

        model_params = {"transition": transition_prob, "emission": emission_prob}

        with open(file_path, "w") as json_file:
            json.dump(model_params, json_file, indent=4)
