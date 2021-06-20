"""Testing the data available from 70propred paper"""

print("starting...")
import os
import sys
import pandas as pd
import numpy as np
import argparse
import re
import random

import matplotlib.pyplot as plt
import requests

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import itertools
import tensorflow.keras.backend as K
from sklearn.externals import joblib


MODEL_OUTFILE = "data/Training/Model/CNN_model.h5"


def Anne_one_hot_encode(seq):
    mapping = dict(zip("ACGT", range(4)))
    seq2 = [mapping[i] for i in seq]
    return np.eye(4)[seq2]


def getSequences(sequences):
    output = []
    temp = []
    for i in sequences:
        if len(i) != 3:
            if i == ['']:
                i = []
            i = temp+i
            temp = i
        if len(i) == 3:
            temp = []
            output.append(i[1]+i[2])

    return output

promoter_file = open("70pred data/promoters.txt")
promoters = promoter_file.readlines()
x = (promoter.strip().split(" ") for promoter in promoters)
promoter_list = getSequences(x)

print('Number of Promoters: ' + str(len(promoter_list)))

non_promoter_file = open("70pred data/nonpromoters.txt")
non_promoters = non_promoter_file.readlines()
x = (nonpromoter.strip().split(" ") for nonpromoter in non_promoters)
non_promoter_list = getSequences(x)

print('Number of Non-Promoters: ' + str(len(non_promoter_list)))


print('Training the model...')

# 1. Seperate out a test set
test_proportion = 0.1
print('Using ' +str(test_proportion*100) +' percent of the data for the test set\n')

counter_for_data_split = 100 / (100 * test_proportion)

training_sequences = []
training_response = []
test_sequences = []
test_response = []

for i in range (0,len(promoter_list)):
    if i % counter_for_data_split == 0:
        test_sequences.append(promoter_list[i])
        test_response.append(1)
    else:
        training_sequences.append(promoter_list[i])
        training_response.append(1)


for i in range (0,len(non_promoter_list)):
    if i % counter_for_data_split == 0:
        test_sequences.append(non_promoter_list[i])
        test_response.append(0)
    else:
        training_sequences.append(non_promoter_list[i])
        training_response.append(0)



# 2. Get sequences ready for training as features

# The LabelEncoder encodes a sequence of bases as a sequence of integers.
integer_encoder = LabelEncoder()
# The OneHotEncoder converts an array of integers to a sparse matrix where
# each row corresponds to one possible value of each feature.
one_hot_encoder = OneHotEncoder(categories='auto')

train_features = []
for sequence in training_sequences:
	train_features.append(Anne_one_hot_encode(sequence))

np.set_printoptions(threshold=40)
train_features = np.stack(train_features)

test_features = []
for sequence in test_sequences:
	test_features.append(Anne_one_hot_encode(sequence))

np.set_printoptions(threshold=40)
test_features = np.stack(test_features)

# 3. Get Responses ready for training as labels

train_labels = training_response
one_hot_encoder = OneHotEncoder(categories='auto')
train_labels = np.array(train_labels).reshape(-1, 1)
train_labels = one_hot_encoder.fit_transform(train_labels).toarray()

test_labels = test_response
one_hot_encoder = OneHotEncoder(categories='auto')
test_labels = np.array(test_labels).reshape(-1, 1)
test_labels = one_hot_encoder.fit_transform(test_labels).toarray()



# 4. selecting the architecture

model = Sequential()
model.add(Conv1D(filters=32, kernel_size=12,
                 input_shape=(train_features.shape[1], 4)))
model.add(MaxPooling1D(pool_size=4))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['binary_accuracy'])
print(model.summary())


# 5. training the network
# training set further divided into training and validation set

history = model.fit(train_features, train_labels,
                   epochs=50, verbose=0, validation_split=0.25)

# 6. Save the model
model.save(MODEL_OUTFILE)

# 7. Validation using test set
predicted_labels = model.predict(np.stack(test_features))
cm = confusion_matrix(np.argmax(test_labels, axis=1),
                      np.argmax(predicted_labels, axis=1))
print('Confusion matrix:\n',cm)

# ''' OUTPUT TEST PREDICTIONS FOR FURTHER INVESTIGATION '''
# correct_predictions = []
# false_predictions = []
#
# for i in range(0, len(test_features)):
#     if test_labels[i][1] == 1 and round(predicted_labels[i][1]) == 1:
#         correct_predictions.append(test_features[i])
#     if test_labels[i][1] == 1 and round(predicted_labels[i][1]) == 0:
#         false_predictions.append(test_features[i])
#
# ID_correct = []
# value = 1
# for i in correct_predictions:
#     if i in promoter_list:
#         header = promoter_list.index(i) - 1
#         ID_correct.append(promoter_list[header])
#     else:
#         ID_correct.append('>NONPROMOTER|'+str(value))
#         value +=1
#
#
# ID_false = []
# value = 1
# for i in false_predictions:
#     if i in promoter_list:
#         header = promoter_list.index(i) - 1
#         ID_false.append(promoter_list[header])
#     else:
#         ID_false.append('>NONPROMOTER|'+str(value))
#         value +=1
#
#
# # correct_file_loc = open('data/test_predictions/correct_predictions.txt', 'w')
# # false_file_loc = open('data/test_predictions/false_predictions.txt', 'w')
#
# for i in range(0, len(correct_predictions)):
#     correct_file_loc.write(str(ID_correct[i]))
#     correct_file_loc.write('\n')
#     correct_file_loc.write(str(correct_predictions[i]))
#     correct_file_loc.write('\n')
# correct_file_loc.close()
#
# for i in range(0, len(false_predictions)):
#     false_file_loc.write(str(ID_false[i]))
#     false_file_loc.write('\n')
#     false_file_loc.write(str(false_predictions[i]))
#     false_file_loc.write('\n')
# false_file_loc.close()
