import itertools
import json
import os
import warnings

warnings.filterwarnings("ignore")

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
    GREEDY_ALGO_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "greedy.json")
    VITERBI_ALGO_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "viterbi.json")


class WSJDatasetConfig:
    cols = ["index", "sentences", "labels"]

    train_file_path = os.path.join(PathConfig.DATA_PATH, "train.json")
    dev_file_path = os.path.join(PathConfig.DATA_PATH, "dev.json")
    test_file_path = os.path.join(PathConfig.DATA_PATH, "test.json")


class VocabConfig:
    UNKNOWN_TOKEN = "<unk>"
    THRESHOLD = 2
    FILE_HEADER = ["word", "index", "frequency"]

    VOCAB_FILE = PathConfig.VOCAB_FILE_PATH


class HMMConfig:
    HMM_MODEL_SAVED = PathConfig.HMM_MODEL_SAVE_PATH


class WSJDataset:
    def __init__(self, path, split="train"):
        self.path = path
        self.split = split

        self.data: pd.DataFrame = None
        self.cols = WSJDatasetConfig.cols

    def _read_data(self):
        self.data = pd.read_json(self.path)
        return self.data

    def _process_sentences(self):
        self.data["sentence"] = self.data["sentence"].apply(
            lambda sentence: [word.lower() for word in sentence],
        )

    def prepare_dataset(self):
        self._read_data()
        self._process_sentences()
        return self.data

    def get_sentences_with_pos_tags(self):
        if "labels" in self.data.columns:
            sentences_with_pos_tags = self.data.loc[:, ["sentence", "labels"]].apply(
                lambda row: list(zip(row["sentence"], row["labels"])), axis=1
            )
        else:
            sentences_with_pos_tags = self.data["sentence"].apply(
                lambda sentence: list(zip(sentence, [None] * len(sentence)))
            )
        sentences_with_pos_tags = sentences_with_pos_tags.tolist()
        return sentences_with_pos_tags


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
        frequencies. It replaces words with frequencies less than the specified threshold with
        the unknown token ("<unk>"). The resulting DataFrame is sorted by frequency and indexed.

        If the 'save' flag is set, the vocabulary will be saved to the specified path.

        Usage:
            ```py
            vocab_generator = VocabularyGenerator(threshold=3, unknown_token="<unk>")
            vocab_df = vocab_generator.generate_vocabulary(data, sentence_col_name)
            ```
        """
        word_freq_df = self._count_word_frequency(data, sentence_col_name)

        # Replace words with frequency less than threshold with '<unk>'
        word_freq_df["word"] = word_freq_df.apply(
            lambda row: self.unknown_token if row["frequency"] <= self.threshold else row["word"],
            axis=1,
        )

        # Group by 'Word' and aggregate by sum
        word_freq_df = word_freq_df.groupby("word", as_index=False)["frequency"].agg("sum")

        # Sort the DataFrame by frequency
        word_freq_df = word_freq_df.sort_values(by="frequency", ascending=False, ignore_index=True)

        # Placing Special Tokens at the top of the DataFrame
        unk_df = word_freq_df.loc[word_freq_df["word"] == self.unknown_token]
        word_freq_df = word_freq_df.loc[word_freq_df["word"] != self.unknown_token]

        word_freq_df = pd.concat([unk_df, word_freq_df], ignore_index=True)

        # Add an index column
        word_freq_df["index"] = range(len(word_freq_df))

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

        # Laplace Smoothing
        self.smoothing_constant = 1e-10

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
        self.priors = np.zeros(num_states)

    def _smoothen_propabilities(self, prob_mat: np.array, smoothing_constant: float):
        """Handle cases where the probabilities is 0"""
        return np.where(prob_mat == 0, smoothing_constant, prob_mat)

    def _compute_prior_params(self, train_data):
        tag_to_index = {tag: i for i, tag in enumerate(self.labels)}
        num_sentences = len(train_data)

        for sentence in train_data:
            label = sentence[0][1]
            state_idx = tag_to_index[label]
            self.priors[state_idx] += 1

        self.priors = self.priors / num_sentences
        self.priors = self._smoothen_propabilities(self.priors, self.smoothing_constant)

    def _compute_transition_params(self, train_data):
        tag_to_index = {tag: i for i, tag in enumerate(self.labels)}

        for sentence in train_data:
            label_indices = [tag_to_index.get(label) for _, label in sentence]

            for i in range(1, len(label_indices)):
                prev_state = label_indices[i - 1]
                curr_state = label_indices[i]
                self.transitions[prev_state, curr_state] += 1

        row_agg = self.transitions.sum(axis=1)[:, np.newaxis]
        self.transitions = self.transitions / row_agg
        self.transitions = self._smoothen_propabilities(self.transitions, self.smoothing_constant)

    def _compute_emission_params(self, train_data):
        word_to_index = dict(zip(self.vocab["word"], self.vocab["index"]))
        tag_to_index = {tag: i for i, tag in enumerate(self.labels)}

        for sentence in train_data:
            for word, label in sentence:
                state_idx = tag_to_index[label]
                word_idx = word_to_index.get(word, word_to_index[VocabConfig.UNKNOWN_TOKEN])
                self.emissions[state_idx, word_idx] += 1

        row_agg = self.emissions.sum(axis=1)[:, np.newaxis]
        self.emissions = self.emissions / row_agg
        self.emissions = self._smoothen_propabilities(self.emissions, self.smoothing_constant)

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
            f"({s1}, {s2})": self.transitions[self.states.index(s1), self.states.index(s2)]
            for s1, s2 in itertools.product(self.states, repeat=2)
        }

        emission_prob = {
            f"({s}, {w})": p
            for s in self.states
            for w, p in zip(self.vocab["word"], self.emissions[self.states.index(s), :])
        }

        model_params = {"transition": transition_prob, "emission": emission_prob}

        with open(file_path, "w") as json_file:
            json.dump(model_params, json_file, indent=4)


class GreedyDecoding:
    def __init__(self, prior_probs, transition_probs, emission_probs, states, vocab):
        self.priors = prior_probs
        self.transitions = transition_probs
        self.emissions = emission_probs
        self.states = states
        self.vocab = vocab

        self.tag_to_idx = {tag: idx for idx, tag in enumerate(states)}
        self.word_to_index = dict(zip(self.vocab["word"], self.vocab["index"]))

        # Precompute scores for each word-tag pair
        self.priors_emissions = prior_probs[:, np.newaxis] * emission_probs

    def _decode_single_sentence(self, sentence):
        predicted_tags = []

        prev_tag_idx = None

        for word in sentence:
            word_idx = self.word_to_index.get(word, self.word_to_index[VocabConfig.UNKNOWN_TOKEN])

            if prev_tag_idx is None:
                # scores = self.priors * self.emissions[:, word_idx]
                scores = self.priors_emissions[:, word_idx]
            else:
                scores = self.transitions[prev_tag_idx] * self.emissions[:, word_idx]

            prev_tag_idx = np.argmax(scores)
            predicted_tags.append(self.states[prev_tag_idx])

        return predicted_tags

    def decode(self, sentences):
        predicted_tags_list = []

        for sentence in sentences:
            predicted_tags = self._decode_single_sentence([word for word, tag in sentence])
            predicted_tags_list.append(predicted_tags)

        return predicted_tags_list


class ViterbiDecoding:
    def __init__(self, prior_probs, transition_probs, emission_probs, states, vocab):
        self.priors = prior_probs
        self.transitions = transition_probs
        self.emissions = emission_probs
        self.states = states
        self.vocab = vocab

        self.num_states = len(self.states)

        # Index Conversion dictionary for mapping
        self.tag_to_idx = {tag: idx for idx, tag in enumerate(states)}
        self.word_to_idx = dict(zip(self.vocab["word"], self.vocab["index"]))

        # Precompute scores for each word-tag pair
        self.priors_emissions = prior_probs[:, np.newaxis] * emission_probs

    def _initialize_variables(self, sentence):
        V = np.zeros((len(sentence), self.num_states))
        path = np.zeros((len(sentence), self.num_states), dtype=int)

        word_idx = np.array(
            [
                self.word_to_idx.get(word, self.word_to_idx[VocabConfig.UNKNOWN_TOKEN])
                for word in sentence
            ]
        )

        return V, path, word_idx

    def _decode_single_sentence(self, sentence):
        V, path, word_idx = self._initialize_variables(sentence)

        V[0] = np.log(self.priors_emissions[:, word_idx[0]])

        for t in range(1, len(sentence)):
            # Compute scores
            scores = (
                V[t - 1, :, np.newaxis]
                + np.log(self.transitions)
                + np.log(self.emissions[:, word_idx[t]])
            )
            V[t] = np.max(scores, axis=0)
            path[t] = np.argmax(scores, axis=0)

        # Backtracking
        predicted_tags = [0] * len(sentence)
        predicted_tags[-1] = np.argmax(V[-1])

        for t in range(len(sentence) - 2, -1, -1):
            predicted_tags[t] = path[t + 1, predicted_tags[t + 1]]

        predicted_tags = [self.states[tag_idx] for tag_idx in predicted_tags]
        return predicted_tags

    def decode(self, sentences):
        predicted_tags_list = []

        for sentence in sentences:
            predicted_tags = self._decode_single_sentence([word for word, tag in sentence])
            predicted_tags_list.append(predicted_tags)

        return predicted_tags_list


def calculate_accuracy(predicted_sequences, true_sequences):
    """
    Calculate the accuracy of predicted sequences compared to true sequences.

    Args:
        predicted_sequences (list): List of predicted sequences.
        true_sequences (list): List of true sequences.

    Returns:
        float: Accuracy as a percentage.
    """
    total = 0
    correct = 0

    for true_label, predicted_label in zip(true_sequences, predicted_sequences):
        for true_tag, predicted_tag in zip(true_label, predicted_label):
            total += 1
            if true_tag == predicted_tag:
                correct += 1

    accuracy = correct / total
    return accuracy


def train_and_evaluate():
    train_dataset = WSJDataset(path=WSJDatasetConfig.train_file_path)
    df_train = train_dataset.prepare_dataset()

    valid_dataset = WSJDataset(path=WSJDatasetConfig.dev_file_path)
    df_valid = valid_dataset.prepare_dataset()

    unp_test_df = WSJDataset(path=WSJDatasetConfig.test_file_path)._read_data()

    test_dataset = WSJDataset(path=WSJDatasetConfig.test_file_path)
    test_dataset.prepare_dataset()

    vocab_generator = VocabularyGenerator(
        threshold=VocabConfig.THRESHOLD, unknown_token=VocabConfig.UNKNOWN_TOKEN, save=True
    )
    vocab_df = vocab_generator.generate_vocabulary(df_train, "sentence")
    print("Selected threshold for unknown words: ", VocabConfig.THRESHOLD)
    print("Vocabulary size: ", vocab_df.shape[0])
    print(
        "Total occurrences of the special token <unk>: ",
        int(vocab_df[vocab_df["word"] == "<unk>"].frequency),
    )

    unique_pos_tags = df_train.labels.explode().unique()
    unique_pos_tags = unique_pos_tags.tolist()

    train_sentences_with_pos_tags = train_dataset.get_sentences_with_pos_tags()
    valid_sentences_with_pos_tags = valid_dataset.get_sentences_with_pos_tags()
    test_sentences_with_pos_tags = test_dataset.get_sentences_with_pos_tags()

    model = HMM(vocab_file=VocabConfig.VOCAB_FILE, labels=unique_pos_tags)
    model.fit(train_sentences_with_pos_tags)
    model.save_model()

    p, t, e = model.get_all_probability_matrices
    print("Number of Transition Parameters =", len(t.flatten()))
    print("Number of Emission Parameters =", len(e.flatten()))

    # Assuming you have the probability matrices and other data
    greedy_decoder = GreedyDecoding(p, t, e, model.states, model.vocab)

    # Apply Greedy Decoding on development data
    predicted_dev_tags = greedy_decoder.decode(valid_sentences_with_pos_tags)

    # Apply Greedy Decoding on Test data
    predicted_test_tags = greedy_decoder.decode(test_sentences_with_pos_tags)

    acc = calculate_accuracy(predicted_dev_tags, df_valid.labels.tolist())
    print("Greedy Decoding Accuracy: ", round(acc, 4))

    df_greedy_preds = unp_test_df.copy(deep=True)
    df_greedy_preds["labels"] = predicted_test_tags

    df_greedy_preds.to_json(PathConfig.GREEDY_ALGO_OUTPUT_PATH, orient="records", indent=4)

    # Assuming you have the probability matrices and other data
    viterbi_decoder = ViterbiDecoding(p, t, e, model.states, model.vocab)

    # Apply Greedy Decoding on development data
    predicted_dev_tags_viterbi = viterbi_decoder.decode(valid_sentences_with_pos_tags)

    acc_v = calculate_accuracy(predicted_dev_tags_viterbi, df_valid.labels.tolist())
    print("Viterbi Decoding Accuracy: ", round(acc_v, 4))

    # Apply Greedy Decoding on Test data
    predicted_test_tags_v = greedy_decoder.decode(test_sentences_with_pos_tags)

    df_viterbi_preds = unp_test_df.copy(deep=True)
    df_viterbi_preds["labels"] = predicted_test_tags_v

    df_viterbi_preds.to_json(PathConfig.VITERBI_ALGO_OUTPUT_PATH, orient="records", indent=4)


if __name__ == "__main__":
    train_and_evaluate()
