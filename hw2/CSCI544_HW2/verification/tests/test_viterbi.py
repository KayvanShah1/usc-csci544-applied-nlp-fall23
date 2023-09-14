import json
from pathlib import Path


def test_viterbi_exists(viterbi):
    """This test case checks if the viterbi file exists"""
    assert Path(viterbi).exists(), "viterbi file was not found at the specified location"


def test_viterbi_is_json(viterbi):
    """This test case checks if the viterbi file is a valid json"""
    try:
        with open(viterbi) as f:
            json.load(f)
    except ValueError as e:
        assert False, "viterbi file is not in proper json format"


def test_viterbi_num_records(viterbi, test):
    with open(viterbi) as f:
        predictions_json = json.load(f)
    
    with open(test) as f:
        test_json = json.load(f)

    assert len(predictions_json) == len(test_json)


def test_viterbi_records_format(viterbi):
    """This test checks if all the records in the viterbi file have the attributes: index, sentence, labels"""
    with open(viterbi) as f:
        predictions_json = json.load(f)
    
    for record in predictions_json:
        assert len(set(record.keys()).intersection({'index', 'sentence', 'labels'})) == 3, "The items in your viterbi.json file should have index, sentence, labels. No more, no less"


def test_viterbi_sentence_len(viterbi, test):
    """This test checks if all the records in the viterbi file have sentence length equal to the test file"""
    with open(viterbi) as f:
        predictions_json = json.load(f)
    
    with open(test) as f:
        test_json = json.load(f)
    
    for predictions_record, test_record in zip(predictions_json, test_json):
        assert len(predictions_record['sentence']) == len(test_record['sentence']), f"sentence at {test_record['index']} from test.json has a different size than your file"


def test_viterbi_sentences_labels_equal_len(viterbi):
    """This test checks if all the records in the viterbi file have the sentence length equal to the label length"""
    with open(viterbi) as f:
        predicitions_json = json.load(f)
    
    for record in predicitions_json:
        assert len(record['sentence']) == len(record['labels']), f"sentence at {record['index']} has different number of labels than number of words in the sentence"
       

def test_viterbi_sentence_integrity(viterbi, test):
    """Check if all words are same for a sentence between viterbi.json and test.json"""
    with open(viterbi) as f:
        predictions_json = json.load(f)
    
    with open(test) as f:
        test_json = json.load(f)
    
    for predictions_record, test_record in zip(predictions_json, test_json):
        for pred_word, test_word in zip(predictions_record['sentence'], test_record['sentence']):
            assert pred_word == test_word, f"sentence {predictions_record['index']} in viterbi file has a word mismatch"
