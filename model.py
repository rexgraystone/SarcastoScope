import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gensim
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Embedding, Dense, Dropout, LSTM, GRU
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_json("Sarcasm_Headlines_Dataset_v2.json", lines=True)

num_words = len(df)
dim = 200
stop_words = stopwords.words('english')

def remove_stopwords(X):
    text = []
    for i in X.split():
        if i.strip().lower() not in stop_words:
            text.append(i.strip())
    return text

def weights(model, text):
    size=len(text)+1
    weights=np.zeros((size, dim))
    for word, i in text.items():
        weights[i]=model.wv[word]
    return weights

df['headline'] = df['headline'].apply(remove_stopwords)
X = df['headline']
y = df['is_sarcastic']

word2vec = gensim.models.Word2Vec(sentences=X, vector_size=dim, window=10, min_count=1)

tokenizer = text.Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(X)
train = tokenizer.texts_to_sequences(X)
X = sequence.pad_sequences(train, maxlen=20)
size = len(tokenizer.word_index)+1

vectors = weights(word2vec, tokenizer.word_index)
optimizer = Adam(learning_rate=0.00001)

model = Sequential(
    [
        Embedding(size, output_dim=dim, weights=[vectors], input_length=20),
        LSTM(units=256, recurrent_dropout=0, dropout=0.5, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', unroll=False, use_bias=True),
        LSTM(units=128, recurrent_dropout=0, dropout=0.3, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', unroll=False, use_bias=True),
        GRU(units=64, recurrent_dropout=0, dropout=0.1, activation='tanh', recurrent_activation='sigmoid', unroll=False, use_bias=True),
        Dense(1, activation='sigmoid')
    ])
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

plot_model(model, to_file="Images/model_architecture.png", show_shapes=True, show_layer_names=True, show_layer_activations=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

early_stop = EarlyStopping(patience=10, verbose=1)
history = model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=50, batch_size=64, callbacks=[early_stop], verbose=1)
result = model.evaluate(x=X_test, y=y_test)

metrics = ["accuracy", "loss"]

for metric in metrics:
    plt.clf()
    plt.plot(history.history[metric], label='train')
    plt.plot(history.history[f'val_{metric}'], label='val')
    plt.legend(loc="right")
    plt.xlabel('epochs')
    plt.ylabel(metric)
    plt.title(f"Model {metric.capitalize()}")
    plt.savefig(f'Images/{metric}.png')