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
from keras import optimizers

data_test = pd.read_json('../test.json')
MAX_SEQUENCE_LENGTH = 60

vectors = np.load("../GloVe_codeswitch_5k.npy")
words = np.load('../5k_vocab_dict.npy').item()
EMBEDDING_DIM = len(vectors[0])

nerToIdx = np.load('../ner_to_idx.npy').item()
posToIdx = np.load('../pos_to_idx.npy').item()

NER_NUM = len(nerToIdx.keys())
POS_NUM = len(posToIdx.keys())

switch = "switch"
noswitch = "noswitch"

def createExamplesFull(data):
    examples = []
    labels = []
    num_reviews, review_length = data.shape
    
    toReturn = np.zeros((num_reviews, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM + 3))# + POS_NUM + NER_NUM))

    for r in range(num_reviews):
        
        currExample = np.zeros((MAX_SEQUENCE_LENGTH, EMBEDDING_DIM + 3))# + POS_NUM + NER_NUM))
        label_vec = []

        for w in range(review_length):

            currWordStruct = data[w][r]

            if currWordStruct == None:
                break
                
            currWord = currWordStruct[0]
            currLang = currWordStruct[1]
            currNerE = currWordStruct[4]
            currNerS = currWordStruct[6]
            currPosE = currWordStruct[5]
            currPosS = currWordStruct[3]

            if currWord in words:
                currExample[w,:EMBEDDING_DIM] = vectors[words[currWord],:]
            else:
                currExample[w,:EMBEDDING_DIM] = vectors[words["<UNK>"],:]
                
            if currLang == 'eng':
                currExample[w,EMBEDDING_DIM:EMBEDDING_DIM + 3] = [1, 0, 0]

            elif currLang == 'spa':
                currExample[w,EMBEDDING_DIM:EMBEDDING_DIM + 3] = [0, 1, 0]

            else:
                currExample[w,EMBEDDING_DIM:EMBEDDING_DIM + 3] = [0, 0, 1]

            #currExample[w,EMBEDDING_DIM + 3 + posToIdx[currPosE]] = 1
            #currExample[w,EMBEDDING_DIM + 3 + posToIdx[currPosS]] = 1
            #currExample[w,EMBEDDING_DIM + 3 + POS_NUM + nerToIdx[currNerE]] = 1
            #currExample[w,EMBEDDING_DIM + 3 + POS_NUM + nerToIdx[currNerS]] = 1            

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
        toReturn[r,:,:] = currExample
    
    return toReturn, labels
    
examples_test, labels_test = createExamplesFull(data_test)

le = preprocessing.LabelEncoder()
#le.fit([other, english, spanish])
le.fit([switch, noswitch])
label_transform_test = np.zeros((len(labels_test), MAX_SEQUENCE_LENGTH, 2))
for i, vec in enumerate(labels_test):

    curr = to_categorical([le.transform(vec)], num_classes = 2)[0]
    label_transform_test[i,:len(vec),:] = curr

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
    
    def on_epoch_end(self, epoch, logs={}):
        val_predict = np.argmax((np.asarray(self.model.predict(examples_test))).reshape(-1, 2), axis=1)
        val_targ = np.argmax(label_transform_test.reshape(-1, 2), axis=1)
        print(np.sum(val_predict))
        _val_f1 = f1_score(val_targ, val_predict, average='binary')
        _val_recall = recall_score(val_targ, val_predict, average='binary')
        _val_precision = precision_score(val_targ, val_predict, average='binary')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print("— val_f1: %f — val_precision: %f — val_recall %f" % (_val_f1, _val_precision, _val_recall))
        logs['f1'] = self.val_f1s   	
 
metrics = Metrics()

#embedding_layer = Embedding(len(word_index) + 1,
                            #EMBEDDING_DIM,
                            #weights=[embedding_matrix],
                            #input_length=MAX_SEQUENCE_LENGTH,
                            #trainable=False)

def loss(y_true, y_pred, weights):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

curr_ratio = 0.943
curr_hidden = 800
curr_lr = 0.1

print("Testing ratio", curr_ratio, ", hidden", curr_hidden, ", lr", curr_lr, "\n")	

our_loss = partial(loss, weights=K.variable([curr_ratio, 1]))
sgd = optimizers.SGD(lr=curr_lr, decay=1e-6, momentum=0.9, nesterov=True)

model = Sequential()
model.add(LSTM(curr_hidden, return_sequences=True, name="LSTM", input_shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM + 3)))# + POS_NUM + NER_NUM)))
model.add(TimeDistributed(Dense(2, activation='softmax')))
model.compile(loss=our_loss, optimizer=sgd, metrics=['accuracy'])

model.load_weights('model_weights.h5')

results = model.evaluate(examples_test, label_transform_test)

print(results)

val_predict = np.argmax((np.asarray(model.predict(examples_test))).reshape(-1, 2), axis=1)
val_targ = np.argmax(label_transform_test.reshape(-1, 2), axis=1)
print(np.sum(val_predict))
_val_f1 = f1_score(val_targ, val_predict, average='binary')
_val_recall = recall_score(val_targ, val_predict, average='binary')
_val_precision = precision_score(val_targ, val_predict, average='binary')
print("— test_f1: %f — test_precision: %f — test_recall %f" % (_val_f1, _val_precision, _val_recall))

pred = model.predict(examples_test)
for i in range(20):
	toPrint = []
	for w in range(50):
		if data_test[w][i] is None:
			break		

		arg1 = data_test[w][i][0:2]
		toPrint.append(arg1)		

	print(toPrint)
	print("OUR PREDICTIONS:")
	print(np.argmax(pred[i], axis=1)[:len(toPrint)])
	print("GROUND TRUTH:")
	print(np.argmax(label_transform_test[i], axis=1)[:len(toPrint)]) 


