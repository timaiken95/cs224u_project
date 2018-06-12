import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing
from keras.utils import to_categorical
from keras.layers import *
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.callbacks import Callback
import keras.backend as K
from functools import partial
from itertools import product

data_train = pd.read_json('train.json')
data_val = pd.read_json('val.json')
MAX_SEQUENCE_LENGTH = data_train.shape[1]

vectors = np.load("GloVe_codeswitch_5k.npy")
words = np.load('5k_vocab_dict.npy').item()
EMBEDDING_DIM = len(vectors[0])

switch = "switch"
noswitch = "noswitch"

def createExamplesBinary(data):
    examples = []
    labels = []
    num_reviews, review_length = data.shape

    for r in range(num_reviews):
        review_string = ""
        label_vec = []

        for w in range(review_length):

            currWordStruct = data[w][r]

            if currWordStruct == None:
                break
                
            currWord = currWordStruct[0]
            currLang = currWordStruct[1]

            if currWord in words:
                review_string += (" " + currWord)
            else:
                review_string += (" <UNK>")

            if w < (review_length - 1):
                nextWordStruct = data[w + 1][r]
                if nextWordStruct:

                    nextWord = nextWordStruct[0]
                    nextLang = nextWordStruct[1]

                    if currLang != nextLang:
                        label_vec.append(switch)

                    else:
                        label_vec.append(noswitch)

                else:
                    label_vec.append(noswitch)

        labels.append(label_vec)
        examples.append(review_string)
    
    return examples, labels

examples_train, labels_train = createExamplesBinary(data_train)
examples_val, labels_val = createExamplesBinary(data_val)

tokenizer = Tokenizer(num_words=len(vectors), filters="", lower=False)
tokenizer.fit_on_texts(examples_train)
sequences_train = tokenizer.texts_to_sequences(examples_train)
sequences_val = tokenizer.texts_to_sequences(examples_val)

word_index = tokenizer.word_index
train_data = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
val_data = pad_sequences(sequences_val, maxlen=MAX_SEQUENCE_LENGTH)

embedding_dict = {}
for k,v in words.items():
    embedding_dict[k] = vectors[v]

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embedding_dict.get(word)
    embedding_matrix[i] = embedding_vector

le = preprocessing.LabelEncoder()
le.fit([switch, noswitch])
label_transform_train = np.zeros((len(labels_train), MAX_SEQUENCE_LENGTH, 2))
for i, vec in enumerate(labels_train):

    curr = to_categorical(pad_sequences([le.transform(vec)], maxlen=MAX_SEQUENCE_LENGTH), num_classes = 2)[0]
    label_transform_train[i,:,:] = curr

label_transform_val = np.zeros((len(labels_val), MAX_SEQUENCE_LENGTH, 2))
for i, vec in enumerate(labels_val):

    curr = to_categorical(pad_sequences([le.transform(vec)], maxlen=MAX_SEQUENCE_LENGTH), num_classes = 2)[0]
    label_transform_val[i,:,:] = curr

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
    
    def on_epoch_end(self, epoch, logs={}):
        val_predict = np.argmax((np.asarray(self.model.predict(val_data))).reshape(-1, 2), axis=1)
        val_targ = np.argmax(label_transform_val.reshape(-1, 2), axis=1)
        print(np.sum(val_predict))
        _val_f1 = f1_score(val_targ, val_predict, average='binary')
        _val_recall = recall_score(val_targ, val_predict, average='binary')
        _val_precision = precision_score(val_targ, val_predict, average='binary')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        logs['f1'] = self.val_f1s

        print("— val_f1: %f — val_precision: %f — val_recall %f" % (_val_f1, _val_precision, _val_recall))
        return
    
metrics = Metrics()

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

def loss(y_true, y_pred, weights):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss



best_ratio = 0
best_hidden = 0
best_f1 = -1

ratios = np.random.uniform(1.5, 3.5, 15)
hiddens = np.array(np.random.uniform(300, 2000, 15)).astype('int')

for i in range(20):

    currRatio = ratios[i]
    currHidden = hiddens[i]

    print("\nTesting hidden:", currHidden, "and ratio", currRatio, "\n")

    our_loss = partial(loss, weights=K.variable([1, currRatio]))
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(currHidden, return_sequences=True, name="LSTM"))
    model.add(TimeDistributed(Dense(2, activation='softmax')))
    model.compile(loss=our_loss, optimizer='adam', metrics=['accuracy'])
    results = model.fit(train_data, label_transform_train, verbose=2, epochs=5, validation_data = (val_data, label_transform_val), batch_size=100, callbacks=[metrics])

    f1 = results.history['f1'][0][-1]
    if f1 > best_f1:
        best_ratio = currRatio
        best_hidden = currHidden
        best_f1 = f1

print("\nBest Ratio:", best_ratio)
print("Best Hidden Dim:", best_hidden)
print("Best F1:", best_f1)
