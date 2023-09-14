import json
from pathlib import Path
import re


def test_vocab_exists(vocab):
    """This test case checks if the vocab file exists"""
    assert Path(vocab).exists(), "vocab.txt was not found at the specified location"


def test_vocab_num_columns(vocab):
    """This test case checks if the vocab file has appropriate number of columns"""
    with open(vocab) as f:
        data = f.read()

    data = re.split(r'\n', data.strip('\n'))
    for index, datum in enumerate(data, start=1):
        assert len(datum.split()) == 3, f"vocab file's {index}th row does not have 3 columns"
        

def test_vocab_types(vocab):
    """This test case checks if the vocab file's columns have appropriate types"""
    with open(vocab) as f:
        data = f.read()

    data = re.split(r'\n', data.strip('\n'))
    for index, datum in enumerate(data, start=1):
        word, word_index, frequency = datum.split()
        assert isinstance(word, str), f"word on vocab file's {index}th row is not a string"
        assert word_index.isnumeric(), f"word's index on vocab file's {index}th row is not an int"
        assert frequency.isnumeric(), f"word's frequency on vocab file's {index}th row is not an int"
