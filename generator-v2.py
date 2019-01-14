import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Embedding, Dense, Dropout
import keras.utils as ku


'''
Simple word level RNN - takes and input file or prior readings and generates a new reading.

Has 3 parts:
1) dataset prep
2) create model
3) generate a text
'''

#1) Dataset prep converts tokenized sentences to equal length padded sentences
# Data is a text file with ~2200 readings, loading it here and transforming to lower case
#Tokenize it to get all the words, then convert the corpus into a flat dataset of sentence sequences
text=(open("readings.txt").read())

print(text[:60])
print(type(text))
stop = input('-->Hit Enter to keep going')

tokenizer = Tokenizer()

def dataset_preparation(text):
    """Prepares a string for ML processing through an LSTM

        Args:
            text: any string - long file

        Returns:
            input_sequences: a dataset of equal length sentence sequences
            predictors: x variables
            labels: y variables
    """

    corpus = text.lower().replace("\n", ".").split(".")
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    print(corpus[:60])
    print(type(corpus), total_words)
    stop = input('-->Hit Enter to keep going')
    #convert the cprpus into a flat dataset of sentence sequences
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    #need to pad the sequences to make their lengths equal
    max_sequence_len = max([len(x) for x in input_sequences])
    print(max_sequence_len)
    stop = input('-->Hit Enter to keep going')  ######ERROR COMES AFTER THIS IN THE DATSET PREPARATION!!!!
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    #create predictors and a label - create N-grams sequence as predictors and the next word of the N-gram as label
    predictors, label = input_sequences[:,:-1], input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes = total_words)


"""
Sentence: "they are showing stars now"
PREDICTORS             | LABEL
they                   | are
they are               | learning
they are showing       | stars
they are showing stars | now
"""

#2) LTSMs are a type of RNN with a cell state through which the network makes adjustments
#Modeling - we build a sequential model here with 2 LTSM layers having 400 units each
#First layer is fed in with the input shape - for the next layer to process the same sequences
#Dropout have dropout layers with 25% droupout to avoid over-fitting - last layer ouputs a one hot encoded vector with the word output
def create_model(predictors, label, max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    model.add(Embedding(total_words, 10, input_length=input_len))
    model.add(LSTM(500))
    model.add(Dropout(0.25))
    model.add(LSTM(500))
    model.add(Dropout(0.25))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(predictors, label, epochs=100, verbose=1)

#3) Generating text - this is the function to predict the next word based on the input words (or seed text).
# We will first tokenize the seed text, pad the sequences and pass into the trained model to get predicted word.
# The multiple predicted words can be appended together to get predicted sequence.

def generate_text(seed_text, next_words, max_sequence_len, model):
    for j in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word=""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text


#4) Train the Model using the Readings corpus
X, Y, max_len, total_words = dataset_preparation(text)
model = create_model(X,Y, max_len, total_words)

#5) Show the output and save it
txt = generate_text("libra", 7, 9, model)
txt2 = generate_text("Aries", 7, 9, model)
print(txt, txt2)
text_file = open("Output2.txt", "w")
text_file.write("First sample: %s" % txt)
text_file.close()