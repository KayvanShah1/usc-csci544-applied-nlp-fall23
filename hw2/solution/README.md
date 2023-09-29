# HMM with greedy and viterbi decoding
- Python Version: `3.10.10`

## Directory Tree
```
CSCI544_HW2
├───output
|   ├───hmm.json
|   ├───greedy.json
|   ├───viterbi.json
|   └───vocab.txt
├───data
|   ├───train.json
|   ├───dev.json
|   └───test.json
├───CSCI544_HW2.py
├───README.md
└───requirements.txt
```
- Place the `data` folder at the same level as the `CSCI544_HW2.py` solution file.
- On running the script the outputs files will be generated in `output` folder.

## Execute the script
1. Install the dependencies
    ```bash
    pip install -r requirements.txt
    ```
2. Run the script
    ```bash
    python CSCI544_HW2.py
    ```

### Expected Output
```json
Reading and preparing data ...

Generating vocabulary ...
Saved vocabulary to file hw2\solution\output\vocab.txt
Selected threshold for unknown words:  2
Vocabulary size:  15568
Total occurrences of the special token <unk>:  28581

Training the HMM model ...
Number of Transition Parameters = 2025
Number of Emission Parameters = 700560
Saving model to hw2\solution\output\hmm.json

Validating on dev data and producing inference results for test data ...

Greedy Decoding Accuracy:  0.9155
Saved Greedy Decoding predictions to hw2\solution\output\greedy.json

Viterbi Decoding Accuracy:  0.9323
Saved Viterbi Decoding predictions to hw2\solution\output\viterbi.json
```

## Notes
- Assignment - HW2
- CSCI554: Applied Natural Language Processing 

## Author
- Name: `Kayvan Shah`
- USC ID: `1106650685`
- Contact: 
    - Email: `kpshah@usc.edu`