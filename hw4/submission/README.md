## Run Submission files

To execute the provided Python script, you can follow these steps:

### 1. **Install Required Packages:**
   
   - Make sure you have the necessary packages installed. You can install them using the following command:

   ```bash
   pip install torch tqdm datasets numpy pandas
   ```

### 2. `conlleval` Script:

- Make sure you have the `conlleval` script available. You can download it from [https://www.clips.uantwerpen.be/conll2003/ner/bin/conlleval](https://www.clips.uantwerpen.be/conll2003/ner/bin/conlleval).

- Place the `conlleval` script in the same directory as your `main_script.py`:

    ```plaintext
    submission/
    |-- task1.py
    |-- task2.py
    |-- conlleval.py
    |-- (Other contents of the project)
    ```

### 3. **Check the GloVe Embeddings Loading Section (Task 2):**
- Make sure that you have the saved embeddings file in the specified directory. 
   ```plaintext
    submission/
    |-- embeddings/
    |-- (Other contents of the project)
    ```
- In your script, the model weights path is defined in `PathConfig.GLOVE_100d_File`. Ensure that this directory exists, and the weights file is present within it.

### 4. **Check the Model Loading Section:**
- In the `main` function, verify the loading of the pre-trained model weights.
- Make sure that you have the saved model weights file in the specified directory. 
   ```plaintext
    submission/
    |-- saved_models/
    |-- (Other contents of the project)
    ```
- In your script, the model weights path is defined in `PathConfig.SAVED_MODELS_DIR`. Ensure that this directory exists, and the weights file is present within it.

### 5. **Run the Script:**
   Save the provided script in a file, for example, `main_script.py`. Open a terminal and run the script:

   ```bash
   python task1.py
   python task2.py
   ```

   This will execute the `main` function, which loads the CoNLL 2003 dataset, preprocesses it, trains a BiLSTM model, evaluates it on the validation and test sets, and prints the precision, recall, and F1 score.


# Note
> This should execute the script, load the pre-trained model weights, and print the evaluation results on the validation and test datasets.

> Adjust the paths and filenames based on your specific project structure. If you encounter any issues, double-check the paths and file locations in the script.