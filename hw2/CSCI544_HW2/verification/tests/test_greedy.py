import json
from pathlib import Path


def test_greedy_exists(greedy):
    """This test case checks if the greedy file exists"""
    assert Path(greedy).exists(), "greedy.out was not found at the specified location"


def test_greedy_is_json(greedy):
    """This test case checks if the greedy file is a valid json"""
    try:
        with open(greedy) as f:
            json.load(f)
    except ValueError as e:
        assert False, "greedy.out is not in proper json format"


def test_greedy_num_records(greedy, test):
    with open(greedy) as f:
        predictions_json = json.load(f)
    
    with open(test) as f:
        test_json = json.load(f)

    assert len(predictions_json) == len(test_json)


def test_greedy_records_format(greedy):
    """This test checks if all the records in the greedy file have the attributes: index, sentence, labels"""
    with open(greedy) as f:
        predictions_json = json.load(f)
    
    for record in predictions_json:
        assert len(set(record.keys()).intersection({'index', 'sentence', 'labels'})) == 3, "The items in your greedy.json file should have index, sentence, labels. No more, no less"


def test_greedy_sentence_len(greedy, test):
    """This test checks if all the records in the greedy file have sentence length equal to the test file"""
    with open(greedy) as f:
        predictions_json = json.load(f)
    
    with open(test) as f:
        test_json = json.load(f)
    
    for predictions_record, test_record in zip(predictions_json, test_json):
        assert len(predictions_record['sentence']) == len(test_record['sentence']), f"sentence at {test_record['index']} from test.json has a different size than your file"


def test_greedy_sentences_labels_equal_len(greedy):
    """This test checks if all the records in the greedy file have the sentence length equal to the label length"""
    with open(greedy) as f:
        predicitions_json = json.load(f)
    
    for record in predicitions_json:
        assert len(record['sentence']) == len(record['labels']), f"sentence at {record['index']} has different number of labels than number of words in the sentence"
       

def test_greedy_sentence_integrity(greedy, test):
    """Check if all words are same for a sentence between greedy.json and test.json"""
    with open(greedy) as f:
        predictions_json = json.load(f)
    
    with open(test) as f:
        test_json = json.load(f)
    
    for predictions_record, test_record in zip(predictions_json, test_json):
        for pred_word, test_word in zip(predictions_record['sentence'], test_record['sentence']):
            assert pred_word == test_word, f"sentence {predictions_record['index']} in greedy.out has a word mismatch"

