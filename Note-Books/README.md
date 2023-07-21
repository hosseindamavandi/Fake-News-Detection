# Table of Contents

- [Preprocessed Text Classification Dataset](https://github.com/hosseindamavandi/Fake-News-Detection/tree/main/Note-Books#preprocessingipynb)
- [ANN](#)
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
