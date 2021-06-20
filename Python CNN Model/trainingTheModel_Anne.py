'''
Author: Daniel Kaptijn
Date: 13/02/2020
PyVersion: 3.7.6

Aim: Create a script that does everything up until the CNN model is made (create Frank and train the model):
step one:   Creates a Frank genome based on the size of the input data
step two:   Uses GFF file to extract the promoter sequences from the fna file
step three: Makes non-promoters, sequence based on input genome, amount based on number of promoters
step four:  Trains a CNN model using promoters and non-promoters
step five:  predictes promoters on the test set and outputs two documents, one contains true positives the other contains false negatives (promoters only)

REQUIREMENTS:
Two input files: fna and GFF file of an organism, multiple genomes can be used but make sure they have the same name before extension:
e.g. genome1.fna genome2.fna ; genome1.gff genome2.gff <- will work, however:
genome1.fna genome2.fna ; genome2gff.gff genome1gff.gff <- the gff and fna files do not correspond to one another and will not work.
'''


''' IMPORT MODULES '''

#import os
#import sys
import pandas as pd
import numpy as np
import argparse
import re
import random
#import requests

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential
#from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
#import itertools
#import tensorflow.keras.backend as K
#from sklearn.externals import joblib
#from datetime import datetime


# ---------------------------------------------------------------- parse parameters -------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Train the model')
parser.add_argument('-sessiondir', dest='sessiondir', help='Session Dir', nargs='?', default='.')
parser.add_argument('-genomes', dest='genomes', help='List of Genomes; Name [TAB] type (bed|gff) [TAB] fullpath/filename of .fna and .gff|.bed files')
parser.add_argument('-promlen',dest='PromLen', help='Length of Promoter', nargs='?', default=51)
parser.add_argument('-npr',dest='nonPromRatio', help='Ratio non-promoters / promoters , as interger', nargs='?', default=100)
parser.add_argument('-model', dest='ModelOutfile', help='Output filename of the model', nargs='?', default='CNN_model.h5')
parser.add_argument('-train', dest='TrainOutfile', help='Prefix of theFrank file', nargs='?', default='Train')
parser.add_argument('-frank', dest='FrankFile', help='Output filename prefix for Frank; prefix.fna and prefix.gff', nargs='?', default='Frank')
parser.add_argument('--version', action='version', version='Anne de Jong, version 1.0, Feb 2020')
args = parser.parse_args()


''' Uncomment for local use '''
args.sessiondir = 'G:\\My Drive\\WERK\\PromoterPrediction\\Scripts_Anne\\results'
args.genomes = 'ListOfGenomes.txt'
args.PromLen  = 51
args.FrankFile  = "Frank"
args.TrainOutfile = "Train"
args.ModelOutfile = "CNN_model.h5"
args.nonPromRatio = 100


# Be sure the have real numbers
args.PromLen  = int(args.PromLen)

# The number on non promoters for every promoter
# NOTE, makeManyRandoms is slow; default = 500 which will give for 2000 promtoer 500 x 2000= 1M non-promoters
args.nonPromRatio = int(args.nonPromRatio)


''' --------------------------------------  DEFINE FUNCTIONS ---------------------------------------------------'''

def getCleanSeq(fna_file):
	# get DNA and replace N or n by G or g to prevent error in training; G is considered as most save replacment
	DNA = ''
	with open(fna_file) as lines:
		for line in	lines: 
			if line[0] != ">": 
				DNA += line.strip()
	DNA = re.sub('N', 'G', DNA)
	DNA = re.sub('n', 'g', DNA)
	return DNA	


def randomseq(dnaset,promoters):
	result = ''
	for i in range(args.PromLen):
		r = random.randint(0,len(dnaset)-1)
		result += dnaset[r]
	# makes sure that random sequences will not be the same as a promoter sequence
	if result in promoters: randomseq(dnaset,promoters)
	return result

def makeManyRandoms(count, ACGTcontent, promoters):
	# make 100 bases with proper GC content
	dnaset = "A" * ACGTcontent[0] + "C" *ACGTcontent[1] + "G" * ACGTcontent[2] + "T" * ACGTcontent[3]
	RandomSeq = []
	for i in range(0, count):	RandomSeq.append(randomseq(dnaset, promoters))
	return RandomSeq

def Anne_one_hot_encode(seq):
    mapping = dict(zip("ACGT", range(4)))
    seq2 = [mapping[i] for i in seq]
    return np.eye(4)[seq2]

def reverse_complement(dna):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ''.join([complement[base] for base in dna[::-1]])



def write_log(S):
	# write to console and logfile
	print(S)
	f = open(args.sessiondir+'/'+"log_trainingTheModel.log", "a")
	f.write(S + '\n')
	f.close()	


def Anne_getPercentage(genome):
	# return a list of percentages of A,C,T,G
	percentageList = []
	for base in ['A','C','G','T']:	percentageList.append(round(( genome.count(base) / len(genome) )*100))
	return(percentageList)

def getPercentage(genome):
	percentageList = []
	counter = 0
	Acounter = 0
	Ccounter = 0
	Gcounter = 0
	Tcounter = 0
	for i in genome:
	    counter += 1
	    if i == 'A':
	        Acounter += 1
	    if i == 'C':
	        Ccounter += 1
	    if i == 'G':
	        Gcounter += 1
	    if i == 'T':
	        Tcounter += 1
	percentageList.append(round((Acounter/counter)*100))
	percentageList.append(round((Ccounter/counter)*100))
	percentageList.append(round((Gcounter/counter)*100))
	percentageList.append(round((Tcounter/counter)*100))
	return(percentageList)




''' ======================================   EXTRACTING PROMOTER SEQUENCES FOR TRAINING ================================================'''
write_log('Extracting Promoter Sequences...')

# Get the lsit of genomes
genomes = pd.read_csv(args.sessiondir+'/'+args.genomes, sep='\t', header=[0], comment="#")


promoter_list = []
TSS_list = []
non_promoter_list = []
gff_header = ["genome","db", "type", "start", "end", "dot1", "strand", "dot2", "description"]
# test index=1
for index, genome in genomes.iterrows():
	write_log(str(index) + ' ' + genome['Name'])
	DNA = getCleanSeq(genome['FilePrefix']+'.fna')
	TSS = pd.read_csv(args.sessiondir+'/'+args.TrainOutfile+'.'+genome['Name']+'.promoters.gff', header=None, sep='\t', comment="#", names=gff_header)
	write_log('Number of TSSs in ' + genome['Name'] + ' = ' + str(len(TSS)))
	   
   # Extract the promoters from these files (this is why the GFFs for each organism had to be saved seperately)
	for i in range(0,len(TSS['start'])):
		promoter_seq = ''
		if TSS['strand'][i] == '+':
			promoter_seq    =  DNA[TSS['start'][i]-args.PromLen : TSS['start'][i]]
		if TSS['strand'][i] == '-':
			promoter_seq    =  reverse_complement(DNA[TSS['start'][i]-1 : TSS['start'][i]+args.PromLen-1])
		promoter_list.append(promoter_seq)

    # Create the sequences based on genome ATGC content
	ACGTcontent = Anne_getPercentage(DNA) # return a list of percentages of A,C,T,G

	# Define how many sequences you want to create (50,000 = amount of bases from a genome like MG1363, as each sequence is 51bp, 50,000*51 = 2,550,000)
	NUMBER_OF_NON_PROMOTERS = round(len(TSS)) * args.nonPromRatio
	sequences = makeManyRandoms(NUMBER_OF_NON_PROMOTERS, ACGTcontent, promoter_list)

	promoter_list[0:10]


	#overlap = set(promoter_list).intersection(set(sequences))
	#print('Overlap Promoters and Non-Promoters:' + str(overlap))

	non_promoter_list += sequences

write_log('Number of promoters:     ' + str(len(promoter_list)))
write_log('Number of non-promoters: ' + str(len(non_promoter_list)))




''' ======================================  TRAIN THE MODEL ================================================'''
write_log('Training the model...')

# 1. Seperate out a test set
write_log('Seperate out a test set...')
test_set_fraction = 0.1
test_set_percentage = 100 / (100 * test_set_fraction)
write_log('Using ' +str(test_set_percentage) +' percent of the data for the test set\n')


training_sequences = []
training_response = []
test_sequences = []
test_response = []


# get test sequences based on percentage of test_set_percentage  and add the '1'  response value
for i in range (0,len(promoter_list)):
    if i % test_set_percentage == 0:
        test_sequences.append(promoter_list[i])
        test_response.append(1)
    else:
        training_sequences.append(promoter_list[i])
        training_response.append(1)

# get the non-promoter fraction and add the '0  response value
for i in range (0,len(non_promoter_list)):
	if i % test_set_percentage == 0:
		test_sequences.append(non_promoter_list[i])
		test_response.append(0)
	else:
		training_sequences.append(non_promoter_list[i])
		training_response.append(0)

write_log('Number in training set: promoters + non-promoters: '+str(len(training_sequences)))
write_log('Number in test set: promoters + non-promoters: '+str(len(test_sequences)))
write_log('===========')


# 2. Get sequences ready for training as features
write_log('Encode sequences to an image...')

# The LabelEncoder encodes a sequence of bases as a sequence of integers.
integer_encoder = LabelEncoder()

# The OneHotEncoder converts an array of integers to a sparse matrix where
# each row corresponds to one possible value of each feature.
one_hot_encoder = OneHotEncoder(categories='auto')

write_log('Convert training sequences to image with one_hot_encode')
train_features = []
for sequence in training_sequences: train_features.append(Anne_one_hot_encode(sequence))
np.set_printoptions(threshold=40)
train_features = np.stack(train_features)

write_log('Convert test sequences to image with one_hot_encode')
test_features = []
for sequence in test_sequences:	test_features.append(Anne_one_hot_encode(sequence))
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
model.add(Conv1D(filters=32, kernel_size=12, input_shape=(train_features.shape[1], 4)))
model.add(MaxPooling1D(pool_size=4))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
print(model.summary())

# 5. training the network
# training set further divided into training and validation set
write_log('Fitting the model')
history = model.fit(train_features, train_labels, epochs=50, verbose=0, validation_split=0.25)

# 6. Save the model
write_log('Model saved to: ' + args.sessiondir +'/'+args.ModelOutfile)
model.save(args.sessiondir +'/'+args.ModelOutfile)

# 7. Validation using test set
predicted_labels = model.predict(test_features)
cm = confusion_matrix(np.argmax(test_labels, axis=1), np.argmax(predicted_labels, axis=1))
write_log('Confusion matrix of test set:\n' + str(cm))
np.savetxt(args.sessiondir+'/Confusion_matrix_test_set.txt', cm, delimiter='\t', fmt='%d')


predicted_labels = model.predict(train_features)
cm = confusion_matrix(np.argmax(train_labels, axis=1), np.argmax(predicted_labels, axis=1))
write_log('Confusion matrix of training set:\n' + str(cm))
np.savetxt(args.sessiondir+'/Confusion_matrix_trained_set.txt', cm, delimiter='\t', fmt='%d')


write_log("============\n|   DONE   |\n============\n")









''' ================================ OUTPUT TEST PREDICTIONS FOR FURTHER INVESTIGATION =========================================='''

correct_predictions = []
false_predictions = []

for i in range(0, len(test_features)):
    if test_labels[i][1] == 1 and round(predicted_labels[i][1]) == 1:
        correct_predictions.append(test_features[i])
    if test_labels[i][1] == 1 and round(predicted_labels[i][1]) == 0:
        false_predictions.append(test_features[i])

ID_correct = []
value = 1
for i in correct_predictions:
    if i in promoter_list:
        header = promoter_list.index(i) - 1
        ID_correct.append(promoter_list[header])
    else:
        ID_correct.append('>NONPROMOTER|'+str(value))
        value +=1


ID_false = []
value = 1
for i in false_predictions:
    if i in promoter_list:
        header = promoter_list.index(i) - 1
        ID_false.append(promoter_list[header])
    else:
        ID_false.append('>NONPROMOTER|'+str(value))
        value +=1


f = open(args.sessiondir+'/correct_predictions.txt', 'w')
for i in range(0, len(correct_predictions)):
    f.write(str(ID_correct[i]))
    f.write('\n')
    f.write(str(correct_predictions[i]))
    f.write('\n')
f.close()

f =   open(args.sessiondir+'/false_predictions.txt', 'w')
for i in range(0, len(false_predictions)):
    f.write(str(ID_false[i]))
    f.write('\n')
    f.write(str(false_predictions[i]))
    f.write('\n')
f.close()


# cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
#
# plt.imshow(cm, cmap=plt.cm.Blues)
# plt.title('Normalized confusion matrix')
# plt.colorbar()
# plt.xlabel('Predicted label')
# plt.ylabel('True label')
# plt.xticks([0, 1]); plt.yticks([0, 1])
# plt.grid('off')
# for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#     plt.text(j, i, format(cm[i, j], '.2f'),
#              horizontalalignment='center',
#              color='white' if cm[i, j] > 0.5 else 'black')
# plt.show()
