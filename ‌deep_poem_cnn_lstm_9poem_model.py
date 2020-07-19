import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model, Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, recurrent, Flatten, Merge
from keras.layers.convolutional import Conv1D,MaxPooling1D
from keras.optimizers import RMSprop
from keras import optimizers
from keras.layers.merge import concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot
from keras.models import model_from_json
import tensorflow as tf
import nltk
import os
import numpy
import random
from nltk.corpus import brown

from nltk.corpus import indian
from pickle import dump
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.evaluate import paired_ttest_kfold_cv

import numpy as np
import itertools
from keras.models import Sequential
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_curve, auc
from sklearn.utils import shuffle
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from scipy import interp
from itertools import cycle

import string
import math
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def modelbuild():
    model = Sequential()
    model.add(keras.layers.InputLayer(input_shape=(15,1)))
    keras.layers.embeddings.Embedding(max_words, 15,input_length=15,trainable=False)
 
    model.add(keras.layers.recurrent.SimpleRNN(units = 100, activation='relu',use_bias=True))
    model.add(keras.layers.Dense(units=1000, input_dim = 2000, activation='sigmoid'))
    model.add(keras.layers.Dense(units=500, input_dim=1000, activation='relu'))
    model.add(keras.layers.Dense(units=9, input_dim=500,activation='softmax'))
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
#CNN_LSTM model works but accuracy is around 15%
def CNN_LSTM_model(length,vocab_size):
    #Channel 1
    inputs1 = Input(shape=(length,))
    embedding1 = Embedding(vocab_size,100)(inputs1)
    drop1 = Dropout(0.4)(embedding1)
    conv1 = Conv1D(filters=16,kernel_size=6,activation='relu')(drop1)
    pool1 = MaxPooling1D(pool_size=4)(conv1)
    drop2 = Dropout(0.4)(pool1)
    conv2 = Conv1D(filters=8,kernel_size=5,activation='relu')(drop2)
    pool2 = MaxPooling1D(pool_size=4)(conv2)
    drop3 = Dropout(0.4)(pool2)
    layer = LSTM(120)(drop3)
    drop4 = Dropout(0.4)(layer)
   # dense1 = Dense(256,activation='relu')(drop4)
    #drop2 = Dropout(0.2)(layer)
    outputs = Dense(9,activation='softmax')(drop4)
    
    model = Model(inputs=inputs1,outputs=outputs)
    return model
def multichannel_model(length,vocab_size):
    #Channel 1
    inputs1 = Input(shape=(length,))
    embedding1 = Embedding(vocab_size,50)(inputs1)
    conv1 = Conv1D(filters=32,kernel_size=4,activation='relu')(embedding1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)
    #Channel 2
    inputs2 = Input(shape=(length,))
    embedding2 = Embedding(vocab_size,50)(inputs2)
    conv2 = Conv1D(filters=32,kernel_size=6,activation='relu')(embedding2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)
    #Channel 3
    inputs3 = Input(shape=(length,))
    embedding3 = Embedding(vocab_size,50)(inputs3)
    conv3 = Conv1D(filters=32,kernel_size=8,activation='relu')(embedding3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling1D(pool_size=2)(drop3)
    flat3 = Flatten()(pool3)

	#merging channels
    merged = concatenate([flat1,flat2,flat3])
    #interpretation
    dense1 = Dense(256,activation='relu')(merged)
    outputs = Dense(9,activation='softmax')(dense1)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    return model
def multichannellstm_model(length,vocab_size):
    #Channel 1
    inputs1 = Input(shape=(length,))
    embedding1 = Embedding(vocab_size,50,input_length=length)(inputs1)
    conv1 = Conv1D(filters=32,kernel_size=4,activation='relu')(embedding1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)
    #Channel 2
    inputs2 = Input(shape=(length,))
    embedding2 = Embedding(vocab_size,50,input_length=length)(inputs2)
    conv2 = Conv1D(filters=32,kernel_size=6,activation='relu')(embedding2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)
    #Channel 3
    inputs3 = Input(shape=(length,))
    embedding3 = Embedding(vocab_size,50,input_length=length)(inputs3)
    conv3 = Conv1D(filters=32,kernel_size=8,activation='relu')(embedding3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling1D(pool_size=2)(drop3)
    flat3 = Flatten()(pool3)

	#merging channels
    
    merged = concatenate([flat1,flat2,flat3])
    print(tf.shape(merged))
   #dense = Dense(256,activation='relu')(merged)
    #tf.reshape(merged,[merged.shape,1])
    layer = LSTM(100,return_sequences=True)(merged)
    #interpretation
    dense1 = Dense(256,activation='relu')(layer)
    outputs = Dense(9,activation='softmax')(dense1)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    return model
def sharedIP_cnn(length,vocab_size):
    inputs = Input(shape=(length,))
    embedding = Embedding(vocab_size,100)(inputs)
    conv1 = Conv1D(32, kernel_size=3, activation='relu')(embedding)
    drop1 = Dropout(0.7)(conv1)
    pool1 = MaxPooling1D(pool_size=(2))(drop1)
    flat1 = Flatten()(pool1)
# second feature extractor
    conv2 = Conv1D(32, kernel_size=4, activation='relu')(embedding)
    drop2 = Dropout(0.7)(conv2)
    pool2 = MaxPooling1D(pool_size=(2))(drop2)
    flat2 = Flatten()(pool2)
# merge feature extractors
    conv3 = Conv1D(32, kernel_size=5, activation='relu')(embedding)
    drop3 = Dropout(0.7)(conv3)
    pool3 = MaxPooling1D(pool_size=(2))(drop3)
    flat3 = Flatten()(pool3)
    merge = concatenate([flat1, flat2,flat3])
# interpretation layer
    hidden1 = Dense(100, activation='relu')(merge)
# prediction output
    output = Dense(9, activation='softmax')(hidden1)
    model = Model(inputs, outputs=output)
    return model
def sharedIP_cnn_lstm(length,vocab_size):
    inputs = Input(shape=(length,))
    embedding = Embedding(vocab_size,100)(inputs)
    conv1 = Conv1D(32, kernel_size=3, activation='relu')(embedding)
    drop1 = Dropout(0.7)(conv1)
    pool1 = MaxPooling1D(pool_size=(2))(drop1)
    flat1 = Flatten()(pool1)
# second feature extractor
    conv2 = Conv1D(32, kernel_size=4, activation='relu')(embedding)
    drop2 = Dropout(0.7)(conv2)
    pool2 = MaxPooling1D(pool_size=(2))(drop2)
    flat2 = Flatten()(pool2)
# third feature extractor    
    lstm1 = LSTM(100)(embedding)
    #layer= recurrent.SimpleRNN(units = 100, activation='relu',use_bias=True)(layer)
    dense1 = Dense(256,name='FC1')(lstm1)
    dense11 = Activation('relu')(dense1)
    drop3 = Dropout(0.8)(dense11)
    #flat3 = Flatten()(drop3)

    merge = concatenate([flat1,flat2,drop3])
# interpretation layer
    hidden1 = Dense(100, activation='relu')(merge)
# prediction output
    output = Dense(9, activation='softmax')(hidden1)
    model = Model(inputs, outputs=output)
    return model

def sharedIP_lstm(length,vocab_size):
    inputs = Input(shape=(length,))
    embedding = Embedding(vocab_size,100)(inputs)
    lstm1 = LSTM(100)(embedding)
    #layer= recurrent.SimpleRNN(units = 100, activation='relu',use_bias=True)(layer)
    dense1 = Dense(256,name='FC1')(lstm1)
    dense11 = Activation('relu')(dense1)
    drop1 = Dropout(0.8)(dense11)
   # flat1 = Flatten()(drop1)
# second feature extractor
    lstm2= LSTM(100)(embedding)
    #layer= recurrent.SimpleRNN(units = 100, activation='relu',use_bias=True)(layer)
    dense2 = Dense(256,name='FC2')(lstm2)
    dense22 = Activation('relu')(dense2)
    drop2 = Dropout(0.8)(dense22)
    #flat2 = Flatten()(drop2)
# merge feature extractors
    
    merge = concatenate([drop1, drop2])
# interpretation layer
    hidden1 = Dense(100, activation='relu')(merge)
# prediction output
    output = Dense(9, activation='softmax')(hidden1)
    model = Model(inputs, outputs=output)
    return model

def cnn_lstm_merged(length,vocab_size):
    inputs = Input(shape=(length,)) 
    cnn_model = Sequential()
    cnn_model.add(Embedding(vocab_size,100,input_length=length))
    cnn_model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    cnn_model.add(MaxPooling1D(pool_size=2))
    cnn_model.add(Flatten())

    lstm_model = Sequential()
    lstm_model.add(Embedding(vocab_size,100,input_length=length))
    lstm_model.add(LSTM(64, activation = 'relu',return_sequences=True))
    lstm_model.add(Flatten())

   # merge = concatenate([lstm_model, cnn_model])
    merge = Merge([lstm_model, cnn_model], mode='concat')
    hidden = Dense(9, activation = 'sigmoid')
    conc_model = Sequential()
    conc_model.add(merge)
    conc_model.add(hidden)
    return conc_model  

def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(100)(layer)
    #layer= recurrent.SimpleRNN(units = 100, activation='relu',use_bias=True)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    
   # model.add(Dense(50, activation='relu'))
   # layer = Dense(50,name='FC2')(layer)
   # layer = Activation('relu')(layer)
   # layer = Dropout(0.5)(layer)
    
    layer = Dense(9,name='out_layer')(layer)
    layer = Activation('softmax')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model 

def RNN1():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    #layer = LSTM(100)(layer)
    layer= recurrent.SimpleRNN(units = 100, activation='relu',use_bias=True)(layer)
    layer = Dropout(0.3)(layer)
    layer = Dense(1000,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    # model.add(Dense(50, activation='relu'))
   # layer = Dense(50,name='FC2')(layer)
   # layer = Activation('relu')(layer)
   # layer = Dropout(0.5)(layer)
    
    layer = Dense(9,name='out_layer')(layer)
    layer = Activation('softmax')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model  
def createList(foldername, fulldir = True, suffix=".jpg"):
    file_list_tmp = os.listdir(foldername)
    #print len(file_list_tmp)
    file_list = []
    if fulldir:
        for item in file_list_tmp:
            if item.endswith(suffix):
                file_list.append(os.path.join(foldername, item))
    else:
        for item in file_list_tmp:
            if item.endswith(suffix):
                file_list.append(item)
    return file_list

file=createList("poem_new/maitri",suffix=".txt")

file1=createList("poem_new/prem",suffix=".txt")

file2=createList("poem_new/vidamban",suffix=".txt")
file3=createList("poem_new/bhakti",suffix=".txt")
file4=createList("poem_new/shrungar",suffix=".txt")
file5=createList("poem_new/motivation",suffix=".txt")
file6=createList("poem_new/badbad",suffix=".txt")
file7=createList("poem_new/gambhir",suffix=".txt")
file8=createList("poem_new/vinodi",suffix=".txt")
#file6=createList("marathi_poems_upd/anger",suffix=".txt")


#file7=createList("marathi_poems_upd/depression",suffix=".txt")
#file8=createList("marathi_poems_upd/peace",suffix=".txt")

stop=open("marathi_stop‌_new_3.txt","r",encoding='utf-8')
words=stop.read()
word=nltk.word_tokenize(words)
print(word)
documents=[]
all_words=[]
for fname in file:
    a=list(nltk.corpus.indian.words(fname))
    b=[]
    for w in a:
        if w not in word:
            b.append(w)
    documents.extend([(b,"maitri")])
    all_words.extend(b)

for fname1 in file1:
    a=list(nltk.corpus.indian.words(fname1))
    b=[]
    for w in a:
        if w not in word:
            b.append(w)
    documents.extend([(b,"prem")])
    all_words.extend(b)
for fname2 in file2:
    a=list(nltk.corpus.indian.words(fname2))
    b=[]
    for w in a:
        if w not in word:
            b.append(w)
    documents.extend([(b,"vidamban")])
    all_words.extend(b)
for fname3 in file3:
    a=list(nltk.corpus.indian.words(fname3))
    b=[]
    for w in a:
        if w not in word:
            b.append(w)
    documents.extend([(b,"bhakti")])
    all_words.extend(b)
for fname4 in file4:
    a=list(nltk.corpus.indian.words(fname4))
    b=[]
    for w in a:
        if w not in word:
            b.append(w)
    documents.extend([(b,"shrungar")])
    all_words.extend(b)
for fname5 in file5:
    a=list(nltk.corpus.indian.words(fname5))
    b=[]
    for w in a:
        if w not in word:
            b.append(w)
    documents.extend([(b,"motivation")])
    all_words.extend(b)

for fname6 in file6:
    a=list(nltk.corpus.indian.words(fname6))
    b=[]
    for w in a:
        if w not in word:
            b.append(w)
    documents.extend([(b,"badbad")])
    all_words.extend(b)
for fname7 in file7:
    a=list(nltk.corpus.indian.words(fname7))
    b=[]
    for w in a:
        if w not in word:
            b.append(w)
    documents.extend([(b,"gambhir")])
    all_words.extend(b)
for fname8 in file8:
    a=list(nltk.corpus.indian.words(fname8))
    b=[]
    for w in a:
        if w not in word:
            b.append(w)
    documents.extend([(b,"vinodi")])
    all_words.extend(b)    
#all_words_new = nltk.FreqDist(all_words)
#print(all_words_new)
#print(len(all_words_new))
#word_features = list(all_words_new)[:1000]

#print("indian hfhsdjfjfjd")
#ws = [w for w in documents]
print("no. of unique words = ",len(np.unique(all_words)))
#exit(0)
    
random.shuffle(documents)
size = int(len(documents) * 0.7)
tags = [tag for (document, tag) in documents]
train_sents = documents[:size]
#print(len(train_sents))
test_sents = documents[size:]
#trainFeatures, trainLabels = transformDataset(train_sents)
#testFeatures, testLabels = transformDataset(test_sents)
corpus=[d for (d,c) in documents]
labels_old=[c for (d,c) in documents]
print(len(labels_old))
#features=tfidf(corpus)

sns.countplot(labels_old)
plt.xlabel('Label')
plt.title('number of poems of each category')
plt.show()
le = LabelEncoder()
le.fit(labels_old)
labels = le.transform(labels_old)
dummy_y = np_utils.to_categorical(labels)

for i in range(10):
    print("old labels ",labels_old[i],"label",labels[i]," encoded label",dummy_y[i])
#exit(0)    
#labels = labels.reshape(-1,1)
#encoder = OneHotEncoder(sparse=False)
#labels = labels.reshape((2008, 1))
#labels=encoder.fit_transform(labels)




x_train,x_test , trainLabels, testLabels = train_test_split(corpus,dummy_y, test_size=0.2, random_state=42,stratify=dummy_y)
print("train len=",len(x_train))
print("test len=",len(x_test))

       

#encoder = LabelBinarizer()
#encoder.fit(trainLabels)
#y_train = np_utils.to_categorical(trainLabels)
#y_test = np_utils.to_categorical(testLabels)


max_words = 2000
max_len = 200

print(x_train[0])
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
num_labels = 9
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(x_train+x_test)
sequences = tok.texts_to_sequences(x_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len,padding='post')
#model = RNN()
model = CNN_LSTM_model(max_len,max_words)
#model = modelbuild()
print(model.summary())
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#[sequences_matrix,sequences_matrix,sequences_matrix],array(trainLabels)
history =model.fit(sequences_matrix,trainLabels,batch_size=32,epochs=20,
          validation_split=0.2,shuffle=True, callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
#shuffle=False,
#,validation_data=(test_sequences_matrix, testLabels),
def plot_model_performance(train_loss, train_acc, train_val_loss, train_val_acc):
    """ Plot model loss and accuracy through epochs. """

    blue= '#34495E'
    green = '#2ECC71'
    orange = '#E23B13'

    # plot model loss
    #fig, (ax1, ax2) = plt.subplots(2, figsize=(9, 7))
    ax1=plt
    ax1.plot(range(1, len(train_loss) + 1), train_loss, blue, linewidth=5, label='training')
    ax1.plot(range(1, len(train_val_loss) + 1), train_val_loss, green, linewidth=5, label='validation')
    ax1.xlabel('# epoch')
    ax1.ylabel('loss')
   # ax1.tick_params('y')
    ax1.grid(True)
    ax1.legend(loc='upper right', shadow=False)
    ax1.title('Model loss through #epochs', color=orange, fontweight='bold')
    ax1.show()   
    # plot model accuracy
    ax2=plt
    ax2.plot(range(1, len(train_acc) + 1), train_acc, blue, linewidth=5, label='training')
    ax2.plot(range(1, len(train_val_acc) + 1), train_val_acc, green, linewidth=5, label='validation')
    ax2.xlabel('# epoch')
    ax2.ylabel('accuracy')
    #ax2.tick_params('y')
    ax2.grid(True)
    ax2.legend(loc='lower right', shadow=False)
    ax2.title('Model accuracy through #epochs', color=orange, fontweight='bold')
    ax2.show()
plot_model_performance(
    train_loss=history.history.get('loss', []),
    train_acc=history.history.get('acc', []),
    train_val_loss=history.history.get('val_loss', []),
    train_val_acc=history.history.get('val_acc', [])
)

from keras.utils import plot_model

plot_model(model, to_file='cnnlstmpoemmodel.png', show_shapes=True)





test_sequences = tok.texts_to_sequences(x_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len,padding='post')

accr = model.evaluate(test_sequences_matrix,testLabels,verbose=1)
y_pred = model.predict(test_sequences_matrix)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
print(y_pred[0])
y_p_new=[]

classes = ['badbad','bhakti','gambhir','maitri','motivation','prem','shrungar','vidamban','vinodi']

for i in range(len(y_pred)):
    ind = 0
    max = y_pred[i][0]
    for j in range(1,len(y_pred[0])):
        if max < y_pred[i][j] :
            max = y_pred[i][j]
            
            ind = j
    
    y_p_new.append(classes[ind])
print(y_p_new[0])
le.fit(labels_old)
y_p = le.transform(y_p_new)
y_p = np_utils.to_categorical(y_p)    
print(y_p)

precision = dict()
recall = dict()
ther = dict()
average_precision = dict()
print(classification_report(testLabels, y_p, target_names=classes))
precision["micro"], recall["micro"], _ = precision_recall_curve(testLabels.ravel(),y_p.ravel())
average_precision["micro"] = average_precision_score(testLabels, y_p,average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))


    
test_new = []
for i in range(len(testLabels)):
    for j in range(9):
        if (testLabels[i][j]==1.0):
            test_new.append(classes[j])
            
cnf_matrix = confusion_matrix(test_new, y_p_new)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=np.unique(tags),title='Confusion matrix, without normalization')





plt.figure()
plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,where='post')
plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2,color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision["micro"]))
plt.show()


           

for i in range(0,len(classes)):
    precision[i], recall[i], ther[i] = precision_recall_curve(testLabels[:, i],
                                                        y_p[:, i])
    average_precision[i] = average_precision_score(testLabels[:, i], y_p[:, i])
#for i in range(0,len(classes)):
#    print("Precision of",classes[i],"=",precision[i])
#    print("Recall of",classes[i],"=",recall[i])
#    print("Threshold of",classes[i],"=",ther[i])
    
# A "micro-average": quantifying score on all classes jointly
#precision["micro"], recall["micro"], _ = precision_recall_curve(testLabels.ravel(),y_p.ravel())
#average_precision["micro"] = average_precision_score(testLabels, y_p,average="micro")
#print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range (0,len(classes)):
    fpr[i], tpr[i], _ = roc_curve(testLabels[:, i], y_p[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
fpr["micro"], tpr["micro"], _ = roc_curve(testLabels.ravel(), y_p.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(len(classes)):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= len(classes)
    


fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
lw = 2
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],label='micro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["micro"]),color='deeppink' , linestyle=':', linewidth=4)
plt.plot(fpr["macro"], tpr["macro"],label='macro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["macro"]),color='navy', linestyle=':', linewidth=4)
colors = cycle(['aqua', 'fuchsia', 'cornflowerblue','red','green','yellow','pink','teal','black'])
for i, color in zip(range(len(classes)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,label='ROC curve of class {0} (area = {1:0.2f})'''.format(classes[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()

# save the model and use it
# serialize model to JSON
model_json = model.to_json()
with open("cnnlstmpoemmodel.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("cnnlstmpoemmodel.h5")
print("Saved model to disk")

# later...

# load json and create model
json_file = open('cnnlstmpoemmodel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("cnnlstmpoemmodel.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
test_sequences = tok.texts_to_sequences(x_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len,padding='post')

accr = model.evaluate([test_sequences_matrix,test_sequences_matrix,test_sequences_matrix],testLabels,verbose=1)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


        

    









































