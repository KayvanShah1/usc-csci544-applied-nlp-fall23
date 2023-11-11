import os

import itertools
from collections import Counter

from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")

from conlleval import evaluate

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset

from dataclasses import dataclass


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class PathConfig:
    # Get the current dir
    CURRENT_DIR = os.path.dirname(__file__)

    SAVED_MODELS_DIR = os.path.join(CURRENT_DIR, "saved_models")


class DatasetConfig:
    # General Info
    name = "conll2003"

    # Processing
    cols_to_drop = ["id", "pos_tags", "chunk_tags"]
    rename_cols = {"ner_tags": "labels"}

    # Preprocessing
    THRESHOLD = 3
    PAD_TOKEN = "<pad>"
    UNKNOWN_TOKEN = "<unk>"
    embedding_size = 100

    # NER Tags list and converter dictionaries
    ner_tag2idx = {
        "O": 0,
        "B-PER": 1,
        "I-PER": 2,
        "B-ORG": 3,
        "I-ORG": 4,
        "B-LOC": 5,
        "I-LOC": 6,
        "B-MISC": 7,
        "I-MISC": 8,
    }
    ner_idx2tag = {v: k for k, v in ner_tag2idx.items()}

    NUM_NER_TAGS = len(ner_tag2idx)
    SPECIAL_TOKEN_TAG = -100


def get_device():
    if torch.cuda.is_available():
        # Check if GPU is available
        return torch.device("cuda")
    else:
        # Use CPU if no GPU or TPU is available
        return torch.device("cpu")


device = get_device()


def generate_word_indexing(dataset, threshold):
    # Count occurences of the words using itertools and Counter
    word_frequency = Counter(itertools.chain(*dataset))

    # Discard words with frequency below threshold
    word_frequency = {word: freq for word, freq in word_frequency.items() if freq >= threshold}

    # Generate indexes
    word2idx = {word: index for index, word in enumerate(word_frequency.keys(), start=2)}

    # Add special tokens
    word2idx[DatasetConfig.PAD_TOKEN] = 0
    word2idx[DatasetConfig.UNKNOWN_TOKEN] = 1

    return word2idx


@dataclass
class DatasetItem:
    embeddings: torch.Tensor
    targets: torch.Tensor
    original_length: int


class NERDatasetCustom(Dataset):
    def __init__(self, dataset, split, tokenizer, embedding_type="custom"):
        self.name = DatasetConfig.name
        self.dataset = dataset[split]
        self.tokenizer = tokenizer

        # Options: [custom, glove, transformer]
        self.embedding_type = embedding_type

    def __len__(self):
        return self.dataset.num_rows

    def tokenize(self, tokens):
        """
        Code to convert all tokens to their respective indexes
        """
        if self.embedding_type == "glove":
            return [
                self.tokenizer.get(token.lower(), self.tokenizer[DatasetConfig.UNKNOWN_TOKEN])
                for token in tokens
            ]
        return [
            self.tokenizer.get(token, self.tokenizer[DatasetConfig.UNKNOWN_TOKEN])
            for token in tokens
        ]

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError

        item = self.dataset[idx]

        item["input_ids"] = self.tokenize(item["tokens"])

        embeddings = item["input_ids"]
        targets = item["labels"]
        seq_len = len(targets)

        return DatasetItem(
            torch.tensor(embeddings, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
            seq_len,
        )


def collate_fn(data: DatasetItem, tokenizer: dict):
    """
    Collate function for handling padding
    """
    embeddings, targets, og_len = [], [], []

    for item in data:
        embeddings.append(item.embeddings)
        targets.append(item.targets)
        og_len.append(item.original_length)

    # Pad the embeddings sequence
    embeddings = nn.utils.rnn.pad_sequence(
        embeddings, batch_first=True, padding_value=tokenizer[DatasetConfig.PAD_TOKEN]
    )
    # Pad the targets sequence
    targets = nn.utils.rnn.pad_sequence(
        targets, batch_first=True, padding_value=DatasetConfig.SPECIAL_TOKEN_TAG
    )

    return {"embeddings": embeddings, "targets": targets, "original_length": og_len}


class BiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        num_tags,
        hidden_size,
        num_layers,
        lstm_output_size,
        dropout_val,
        embeddings_matrix=None,
    ):
        """
        Recurrent Neural Network (RNN) model for sequence data processing.

        Args:
            vocab_size (int): Size of vocabulary
            embedding_dim (int): Dimension of the input features.
            num_tags (int): Number of output classes.
            hidden_size (int): Number of units in the hidden layers.
            num_layers (int): Number of recurrent layers.
            lstm_output_size (int): Size of the output from the LSTM layer.
            dropout_val (float): Dropout probability.
            embeddings_matrix (np.array): Pretrained embeddings matrix. Default is None

        """
        super(BiLSTM, self).__init__()

        # Model Attributes
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Model Layer Definition
        if embeddings_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.from_numpy(embeddings_matrix).float(), freeze=True
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=True
        )

        self.fc = nn.Linear(hidden_size * 2, lstm_output_size)
        self.dropout = nn.Dropout(dropout_val)
        self.elu = nn.ELU(alpha=0.01)
        self.classifier = nn.Linear(lstm_output_size, num_tags)

    def init_hidden(self, batch_size):
        hidden = (
            torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device),
            torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device),
        )
        return hidden

    def forward(self, x, seq_len):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)

        # Embedding Layer
        embeds = self.embedding(x).float()

        # LSTM layer
        packed_embeds = nn.utils.rnn.pack_padded_sequence(
            embeds, seq_len, batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed_embeds, hidden)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        # Apply fully connected layer for final prediction
        out = self.dropout(out)
        out = self.fc(out)
        out = self.elu(out)
        out = self.classifier(out)

        return out


def evaluate_model(model, data_loader, device, model_type="lstm"):
    all_preds = []
    all_labels = []

    model.eval()

    with torch.no_grad():
        for batch in tqdm(data_loader):
            embeddings = batch["embeddings"].to(device, dtype=torch.long, non_blocking=True)
            labels = batch["targets"].to(device, dtype=torch.long, non_blocking=True)
            seq_lengths = batch["original_length"]

            if model_type == "lstm":
                outputs = model(embeddings, seq_lengths)
            elif model_type == "transformer":
                src_key_padding_mask = batch["src_key_padding_mask"].to(
                    device, dtype=torch.float, non_blocking=True
                )
                outputs = model(embeddings, src_key_padding_mask)

            preds = torch.argmax(outputs, dim=2)

            preds = preds.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            for pred, label, length in zip(preds, labels, seq_lengths):
                pred = [DatasetConfig.ner_idx2tag.get(p, "O") for p in pred[:length]]
                label = [DatasetConfig.ner_idx2tag.get(lb, "O") for lb in label[:length]]
                all_preds.append(pred)
                all_labels.append(label)

    # Evaluate using conlleval
    precision, recall, f1 = evaluate(itertools.chain(*all_labels), itertools.chain(*all_preds))

    return precision, recall, f1


def main():
    dataset = load_dataset("conll2003")
    dataset = dataset.remove_columns(DatasetConfig.cols_to_drop)
    for old_name, new_name in DatasetConfig.rename_cols.items():
        dataset = dataset.rename_column(old_name, new_name)

    word2idx = generate_word_indexing(dataset["train"]["tokens"], threshold=DatasetConfig.THRESHOLD)

    VALID_BATCH_SIZE = 64
    TEST_BATCH_SIZE = 32

    valid_dataset = NERDatasetCustom(
        dataset=dataset,
        split="validation",
        tokenizer=word2idx,
        embedding_type="default",
    )

    test_dataset = NERDatasetCustom(
        dataset=dataset,
        split="test",
        tokenizer=word2idx,
        embedding_type="default",
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=VALID_BATCH_SIZE,
        drop_last=False,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, word2idx),
    )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        drop_last=False,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, word2idx),
    )

    vocab_size = len(word2idx)
    embedding_dim = 100
    hidden_size = 256
    output_size = 128
    num_layers = 1
    dropout_val = 0.33
    num_tags = DatasetConfig.NUM_NER_TAGS

    best_model = BiLSTM(
        vocab_size, embedding_dim, num_tags, hidden_size, num_layers, output_size, dropout_val
    ).to(device)

    # Define the path to the saved weights file
    weights_path = os.path.join(PathConfig.SAVED_MODELS_DIR, "bilstm_custom_embeddings_v7-best.pt")

    # Load the state dict
    state_dict = torch.load(weights_path, map_location=device)

    # Load the state dict into the model
    best_model.load_state_dict(state_dict)

    print("Results on Validation Dataset:")
    precision, recall, f1 = evaluate_model(best_model, valid_data_loader, device)
    print(f"Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1 Score: {f1:.2f}%")

    print("\nResults on Test Dataset:")
    precision, recall, f1 = evaluate_model(best_model, test_data_loader, device)
    print(f"Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1 Score: {f1:.2f}%")


if __name__ == "__main__":
    main()
