#https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/

from numpy import array
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding

#1) LOAD SEQUENCES
# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# load doc and split sentences into lines
in_filename = 'readings_sequences.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')

#2) ENCODE SEQUENCES

#Load Keras Tokenizer and map each word in vocab to a unique integer to encode input sentences
#Later can convert prediction to numbers and look up associated words from vocab
#First train Tokenizer on the entire dataset to find unique words and fit it to encode all the training sequences
#Convert each sequence from a list of words to a list of integers
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)

#Create vocab size, 0 cannot be used so need to add 1
vocab_size = len(tokenizer.word_index) + 1

#3) DEFINE SEQUENCE INPUTS AND OUTPUT, X and Y
#Do this with array slicing, then one-hot encode all words for the network using Keras to_categorical()
#Then specify to the embedding layer how long the input sequences are - we picked 50 in the data-prer portion
#Can also just use the number of dimensions in the column so you don't hardcode 50 here, and we can change the dataprep later
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]

#4) DEFINE AND FIT MODEL TO TRAINING DATA
#Learned embedding needs to know the size of vocab and length of input sequences, alss how many dimensions for each word (size of embedding vector space)
#For LTSMs, we can vary the layers and memory cells - more and deeper is better
#Below is a dense, fully-connected layer with 100 neurons connecting to 2 LSTM hidden layers
#The output latyer predicts the next word as a single vector the size of the vocab with a probability for each word in the vocab
#Softmax activation in the end to normalize the probabilities
model = Sequential()
model.add(Embedding(vocab_size, 200, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())

#5) COMPILE AND FIT MODEL
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X,y, batch_size=512, epochs=50)

#6) SAVE MODEL and TOKENIZER
model.save('model1.h5')
dump(tokenizer, open('tokenizer.pkl', 'wb'))