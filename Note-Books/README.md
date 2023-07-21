# Table of Contents

- [Preprocessed Text Classification Dataset](https://github.com/hosseindamavandi/Fake-News-Detection/blob/main/Note-Books/README.md#preprocessingipynb)
- [Artificial Neural Networks (ANN)](https://github.com/hosseindamavandi/Fake-News-Detection/blob/main/Note-Books/README.md#ann)
- [CNN](#)
- [LSTM](#)


# Preprocessing.ipynb
## Preprocessed Text Classification Dataset

This file contains code for preprocessing a text classification dataset using Python's Natural Language Toolkit (NLTK) and Gensim's Word2Vec model. The dataset used in this code includes both real and fake news samples, and the goal is to prepare the data for machine learning models.

## Requirements

Before running the code, ensure you have the following libraries installed:

- Python (>=3.6)
- NumPy
- Pandas
- NLTK (Natural Language Toolkit)
- Gensim

You can install the required libraries using the following command:

```python
pip install numpy pandas nltk gensim
```

Additionally, download the NLTK resources by running the following Python code:

```python
import nltk
nltk.download('all')
```

## Code Overview
The provided code performs the following tasks:

1. **Data Loading:** The code reads two CSV files, "True.csv" and "Fake.csv," containing real and fake news samples, respectively.
2. **Data Preparation:** The code processes the data by adding a "label" column to indicate real (label=1) and fake (label=0) news. It then combines both datasets into a single dataframe.
3. **Text Preprocessing:** The code tokenizes and preprocesses the text in the "text" column using NLTK. It removes non-alphabetic characters, converts text to lowercase, removes stop words, lemmatizes, and performs stemming.
4. **Word2Vec Model Training:** The code trains a Word2Vec model on the preprocessed tokens from the "text" and "title" columns.
5. **Text Vectorization:** The code defines a function to vectorize text using the trained Word2Vec model. It converts each word in the text to a word vector and then averages them to obtain the text representation.
6. **Vectorization of Columns:** The code applies the vectorization function to both the "text" and "title" columns, creating new columns "text_vector" and "title_vector," respectively.
7. **Data Cleaning:** The code drops rows with missing vector representations to ensure a clean dataset.
8. **Save Preprocessed Data:** The preprocessed data is saved to a CSV file named "preprocessed_dataset.csv" for further use.

## How to Use
1. Ensure you have the required libraries installed and the NLTK resources downloaded.
2. Place the "True.csv" and "Fake.csv" files (containing real and fake news data) in a folder named "datasets" within the project directory.
3. Run the code to preprocess the data and obtain the "preprocessed_dataset.csv" file.


# ANN

This directory contains codes and results related to Artificial Neural Networks (ANN) for Fake News Detection.

## Results

Here are the details and results of different ANN models implemented for Fake News Detection:

1. [ANN.ipynb](https://github.com/hosseindamavandi/Fake-News-Detection/blob/main/Note-Books/ANN/ANN.ipynb):
   - Model Accuracy: 0.9662
   - Details: This notebook demonstrates training a basic ANN model with default parameters.

2. [ANN_Using_L1_L2.ipynb](https://github.com/hosseindamavandi/Fake-News-Detection/blob/main/Note-Books/ANN/ANN_Using_L1_L2.ipynb):
   - Model Accuracy: 0.8964
   - Details: This notebook explores using L1 and L2 regularization techniques with the ANN model.

3. [ANN_Using_Dropout.ipynb](https://github.com/hosseindamavandi/Fake-News-Detection/blob/main/Note-Books/ANN/ANN_Using_Dropout.ipynb):
   - Model Accuracy: :white_check_mark: 0.9696
   - Details: In this notebook, Dropout layers are implemented to prevent overfitting of the ANN model.

4. [ANN_Using_Dropout_L1_L2.ipynb](https://github.com/hosseindamavandi/Fake-News-Detection/blob/main/Note-Books/ANN/ANN_Using_Dropout_L1_L2.ipynb):
   - Model Accuracy: 0.9554
   - Details: This notebook combines Dropout, L1, and L2 regularization techniques to optimize the ANN model's performance.

Each notebook focuses on different aspects of ANN modeling and regularization techniques, providing insights into how to improve the model's accuracy and generalization for Fake News Detection.

The results presented above are based on the specific dataset and configurations used during model training. Keep in mind that results might vary depending on the dataset and specific parameters used.
