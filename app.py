#!/usr/bin/env python
# coding: utf-8


# Keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# Others
import re
import nltk
import string
import pickle
from flask import Flask, render_template, request
import tensorflow as tf

global graph
graph = tf.get_default_graph()
#nltk.data.path.append('nltk_data')

### load tokenizer
vocabulary_size = 5000
tokenizer = pickle.load(open('tokenizer1.pkl', 'rb'))

#And load it back, just to make sure it works:
model = load_model('word_embedding_self2.h5')


 # Clean the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    ## Remove puncuation
    # pu = set(string.punctuation)
    # text = ''.join(ch for ch in text if ch not in pu)
    text = text.translate(str.maketrans(' ',' ',string.punctuation))

    ## Stemming
    text = text.split()
    text = [w for w in text if  len(w) >= 2]
    stemmer = nltk.stem.SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)
    return [text]

maxlen=500

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])


def predict():
    in_text = ''
    in_text = request.form['text']
    if in_text in ['', ' ']:
        return render_template('index.html', prediction_text='Enter valid review')
    text_processed = clean_text(in_text)
    sequence = tokenizer.texts_to_sequences(text_processed)
    pad_sequence = pad_sequences(sequence, maxlen=500)
    with graph.as_default():
        pred_probability = model.predict(x=pad_sequence)
    if pred_probability >= 0.5:
        return render_template('index.html', prediction_text="You liked the movie :)")
    else:
        return render_template('index.html', prediction_text="You didn't like the movie :(")
    
if __name__ == "__main__":
    app.run(debug=True)