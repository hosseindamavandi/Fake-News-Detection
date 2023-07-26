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

def remove_URL(text):
    url = re.compile(r'https?://\S+')
    return url.sub(r' httpsmark ', text)

def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_atsymbol(text):
    name = re.compile(r'@\S+')
    return name.sub(r' atsymbol ', text)

def remove_hashtag(text):
    hashtag = re.compile(r'#')
    return hashtag.sub(r' hashtag ', text)

def remove_exclamation(text):
    exclamation = re.compile(r'!')
    return exclamation.sub(r' exclamation ', text)

def remove_question(text):
    question = re.compile(r'?')
    return question.sub(r' question ', text)

def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r' emoji ', string)

def remove_number(text):
    number = re.compile(r'\d+')
    return number.sub(r' number ', text)

def remove_nulls(data):
    indices = []
    for i in range(len(data)):
        try:
            if len(data.iloc[i]['text_vector']) != 100:
                print(i, len(data.iloc[i]['text_vector']))
        except:
            indices.append(i)

    new_data = data.drop(indices)
    return new_data

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
