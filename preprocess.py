import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec
from nltk.stem import WordNetLemmatizer
import re
import torch

lemma = WordNetLemmatizer()
pattern = "[^a-zA-Z]"


# Tokenize and preprocess text
def preprocess_text(text):
    text = re.sub(pattern, " ", text)  # Cleaning
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    lemmaized_tokens = [lemma.lemmatize(word) for word in filtered_tokens]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in lemmaized_tokens]

    return stemmed_tokens


def vectorize_text(text, model):
    vectors = [model.wv[word] for word in text if word in model.wv]
    if not vectors:
        # If none of the words are in the model's vocabulary, return None
        return None
    # Average the word vectors to get the text representation
    avg_vector = sum(vectors) / len(vectors)
    return avg_vector


def predict(sample, word2vec_model, pytorch_model, device, prepare_for="ANN"):
    pytorch_model.eval()
    processed_sample = preprocess_text(sample)
    processed_sample = vectorize_text(processed_sample, word2vec_model)
    processed_sample = torch.tensor(processed_sample).float().to(device)

    model_output: torch.Tensor = None  # type: ignore
    if prepare_for == "ANN":
        model_output = pytorch_model(processed_sample)
    elif prepare_for == "BILSTM":
        try:
            # * BILSTM Case
            processed_sample = processed_sample.unsqueeze(0).unsqueeze(1)
            print(processed_sample.shape)
            model_output = pytorch_model(processed_sample)
        except:
            print("Change the Value of prepare_for to BILSTM")
            return None
    elif prepare_for == "CNN1D":
        try:
            # * CNN1D Case
            processed_sample = processed_sample.unsqueeze(0)
            print(processed_sample.shape)
            model_output = pytorch_model(processed_sample)
        except:
            print("Change the Value of prepare_for to CNN1D")
            return None
    else:
        print("Set the Value of prepare_for to ANN, CNN1D or BILSTM")

    if model_output > 0.5:
        return f"Real, with probability: {100*model_output.item():.2f}%"
    else:
        return f"Fake, with probability: {100*model_output.item():.2f}%"
