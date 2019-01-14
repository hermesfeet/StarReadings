import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils

'''
Simple character lever RNN - takes and input file or prior readings and generates a new reading.
'''

#Data is a text file with ~2200 readings, loading it here and transforming to lower case
text=(open("readings.txt").read())
text=text.lower()

#Create character mappings - assigning an arbitrary number to each word in the text
char_list = sorted(list(set(text)))
print("Here are some words: ", char_list[1:250])
stop = input('-->Hit Enter to keep going')

n_to_char = {n:char for n, char in enumerate(char_list)}
char_to_n = {char:n for n, char in enumerate(char_list)}

#Data preprocessing1  - X is the train array, Y is the target array
#Seq_length is the legnth of the sequence of words to consider before predicting a particular new word
#For loop iterates over the length of the entire text to create sequences, stored in X, and their final values, ends in Y
X = []
Y = []

length = len(text)
seq_length = 100
for i in range(0, length-seq_length, 1):
    sequence = text[i: i + seq_length]
    label = text[i + seq_length]
    X.append([char_to_n[word] for word in sequence])
    Y.append(char_to_n[label])

#Data preprocessing1  - LTSMS accept inputs in the form of number of sequences, lenght of a sequence, and number of features
#Need to adjust the current array for that, transfor each array Y into a one-hot encoded format
X_modified = np.reshape(X, (len(X), seq_length, 1))
X_modified = X_modified / float(len(char_list))
Y_modified = np_utils.to_categorical(Y)

#Modeling - we build a sequential model here with 2 LTSM layers having 400 units each
#First layer is fed in with the input shape - for the next layer to process the same sequences, we enter the return_sequences parameter as True
#Dropout have dropout layers with 25% droupout to avoid over-fitting - last layer ouputs a one hot encoded vector with the word output
#Crossentropy for the error, Adam as a better optimizer than simple gradient descent
model = Sequential()
model.add(LSTM(400, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM(400))
model.add(Dropout(0.25))
model.add(Dense(Y_modified.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


#Generating text - Start off with a random row from the X array, then target predicting 3 words more following X
#INput is reshaped and scaled as previously.  Seq is used to store the decoded format of the string
string_mapped = X[99]
full_string = [n_to_char[value] for value in string_mapped]
# generating characters
for i in range(seq_length):
    x = np.reshape(string_mapped,(1,len(string_mapped), 1))
    x = x / float(len(char_list))
    pred_index = np.argmax(model.predict(x, verbose=0))
    seq = [n_to_char[value] for value in string_mapped]
    string_mapped.append(pred_index)
    string_mapped = string_mapped[1:len(string_mapped)]

#Combine text and print out the new amount - also save to a local file
txt=""
for char in full_string:
    txt = txt+char
print(txt)
text_file = open("Output.txt", "w")
text_file.write("First sample: %s" % txt)
text_file.close()