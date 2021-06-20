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

print('Starting script...\n')

''' IMPORT MODULES '''
print('Importing modules...')

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

from datetime import datetime
startTime = datetime.now()

print('Done.')
print('=======================')
# ---------------------------------------------------------------- parse parameters -------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Frank maker')
parser.add_argument('-gff', dest='gff_file', help='GFF filename')
parser.add_argument('-fna', dest='fna_file', help='Complete genome FNA filename')
parser.add_argument('-size',dest='FragSize', help='size of fragments', nargs='?', default='100')
parser.add_argument('-frag',dest='FragNum', help='number of fragments', nargs='?', default='4')
parser.add_argument('-out', dest='outfile', help='Output filename prefix; prefix.fna and prefix.gff', nargs='?', default='./')
parser.add_argument('--version', action='version', version='Anne de Jong, version 1.0, Feb 2020')
args = parser.parse_args()


''' GLOBAL VARIABLES '''
# args.FragSize = '100000'
# args.FragNum  = default is 4
args.outfile  = "data/Training/Frank_data/Frank"
TRAIN_OUTFILE = "data/Training/Train_promoters/Train"
MODEL_OUTFILE = "data/Training/Model/CNN_model.h5"

# ---------------------------------------------------------------- For further investigation -------------------------------------------------------------------------------------------
correct_file_loc = open('data/Training/test_predictions/correct_predictions.txt', 'w')
false_file_loc = open('data/Training/test_predictions/false_predictions.txt', 'w')

FILE_ID = []
GFF_PATH = "data/Training/GFF_files/"
FNA_PATH = "data/Training/FNA_files/"
for file in os.listdir(FNA_PATH):
    result = re.match(".+?(?=\.)", file)
    if result:
        FILE_ID.append(result.group(0))
try:
    str(GFF_PATH) + str(FILE_ID)
except:
    print("error: There is no GFF file and a GFF file is required. Input required file at: %s" %(PATH))
    exit()

try:
    str(FNA_PATH) + str(FILE_ID)
except:
    print("error: There is no fna file and an fna file is required. Input required file at: %s" %(PATH))
    exit()

print('gff ='+str([GFF_PATH + i + ".gff" for i in FILE_ID]))
print('fna ='+str([FNA_PATH + i + ".fna" for i in FILE_ID]))
print('File ID ='+str(FILE_ID))
print('FragSize ='+args.FragSize)
print('outfile  ='+args.outfile)
print('model outfile  ='+MODEL_OUTFILE)
print('===========================================================')

PROMOTER_LENGTH = 51
# Define how many sequences you want to create (50,000 = amount of bases from a genome like MG1363, as each sequence is 51bp, 50,000*51 = 2,550,000)
OLD_NUMBER_OF_NON_PROMOTERS = 50000 * 1 # Not used anymore, now defined with non-promoter code further down
# The number on non promoters for every promoter
NON_PROMOTER_RATIO = 100


''' DEFINE FUNCTIONS '''
# ---------------------------------------------------------------- Frank functions -------------------------------------------------------------------------------------------
def random_fragments_from_fna():
	fragments=[]
	for key in fna:
		l=len(fna[key])
		N=int(args.FragNum)
		S=int(args.FragSize)
		for i in range(N):
			start = 1 + round(i*(l/N),0)
			rndstart = random.randint(start,round(start+(l/N)-S,0))
			seq=fna[key][rndstart:rndstart+S]
			record = {'key': key, 'start': rndstart, 'end': rndstart+S, 'seq': seq}
			fragments.append(record)
	return fragments

def load_fna():
	fasta = {}
	key=''
	with open(args.fna_file) as lines:
		for line in	lines:
			items = re.match("^>(.*)", line)
			if items:
				key= items.group(1)
				fasta[key]=''
			else:
				fasta[key]+=line.strip()
	return fasta


def write_fna(header, seq):
	with open(args.outfile+'.fna', 'w') as f:	f.write('>'+header+'\n'+seq)
	f.close()


# ---------------------------------------------------------------- promoter extraction functions ---------------------------------------------------------------------------------------
def remPrevFormatting(fna_file):
    f = open(fna_file)
    output = ''
    for line in f:
        if line[0] != ">":
            output = output + line.strip()

    for i in range(0, len(output)):
        if output[i] == 'N' or output[i] == 'n':
                output = output.replace(output[i], 'G')
    return output


def reverse_complement(dna):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ''.join([complement[base] for base in dna[::-1]])


# ---------------------------------------------------------------- non_promoter functions -------------------------------------------------------------------------------------------
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


def randomseq(A,C,G,T,bp,promoters):
    dnaset = "A" * A + "C" * C + "G" * G + "T" * T
    result = ''
    for i in range(bp):
        r = random.randint(0,len(dnaset)-1)
        result += dnaset[r]
    if result in promoters:
        # makes sure that random sequences will not be the same as a promoter sequence
        randomseq(A,C,G,T,bp,promoters)
    return result


def makeManyRandoms(num, contentList, bp, promoters):
	listOfRandoms = []
	for i in range(0, num):
		listOfRandoms.append(randomseq(contentList[0], contentList[1], contentList[2], contentList[3], bp, promoters))
	return listOfRandoms


# ---------------------------------------------------------------- Training functions -------------------------------------------------------------------------------------------
def Anne_one_hot_encode(seq):
    mapping = dict(zip("ACGT", range(4)))
    seq2 = [mapping[i] for i in seq]
    return np.eye(4)[seq2]


''' RUNNING CODE - MAKING FRANK '''
print('Putting Frank together...')
ConcatFragment='';
GFFlist=[]
GFFtrainlist=[]

for i in range(len(FILE_ID)): # len(GFF_FILES) should always be the same as len(FNA_FILES)
    args.fna_file = FNA_PATH + FILE_ID[i] + ".fna"
    args.gff_file = GFF_PATH + FILE_ID[i] + ".gff"
    try:
        open(args.fna_file)
    except:
        print("input genome is not an fna file as it should be, trying fasta format...")
        try:
            args.fna_file = FNA_PATH + FILE_ID[i] + ".fasta"
            open(args.fna_file)
            print("fasta file loaded successfully, consider converting to fna file for future use.")
        except:
            print("error: fna file not loaded, either in the wrong format or does not have the same name as GFF file, will now exit.")
            exit()
    try:
        open(args.gff_file)
    except:
        print("error: requires GFF file but got something else, will now exit.")
        exit()

    args.FragSize = round(len(remPrevFormatting(args.fna_file)) * (0.1/int(args.FragNum)))
	# 1. Make the fragments
    fna=load_fna()
    fragments=random_fragments_from_fna()

	# 2. Read the nine columns ggf file using pandas
    gff = pd.read_csv(args.gff_file,sep='\t',header=None, dtype=str, comment="#", names=["#genome","db", "type", "start", "end", "dot1", "strand", "dot2", "description"])
    convert_dict = { 'start': int, 'end': int }
    gff = gff.astype(convert_dict)

	# 3. Merge the fragments and the associated gff
    tempGFFlist=[]
    for fragment in fragments:
        print('Fragment: ' + str(fragment['start'])+' - '+str(fragment['end']))
        select = gff[gff['start'].between(fragment['start']+100,fragment['end']-100)]
        select['start'] = select['start'] - fragment['start'] + len(ConcatFragment);
        select['end']   = select['end'] - fragment['start'] + len(ConcatFragment);
        GFFlist.append(select)
        tempGFFlist.append(select)
        ConcatFragment+=fragment['seq']

	# 4. Create the GFF file with above removed for training purposes
    tempTrainGFF = pd.concat(tempGFFlist)
    tempTrainGFF = pd.concat([tempTrainGFF, gff])
    tempTrainGFF.drop_duplicates(subset=["#genome","description"], keep=False, inplace=True)
    GFFtrainlist.append(tempTrainGFF)

# 5. Write Sequence in fasta format and export the new GFF file
write_fna('Frank',ConcatFragment)
pd.concat(GFFlist).to_csv(args.outfile+'.gff', sep='\t', index=False, header=False)

# Deleting the contents of this folder before adding the new files as this folder should only contain information from this run
reset_file = "data/Training/Train_promoters/"
for file in os.listdir(reset_file):
    os.remove(reset_file+file)


# Need to save the TSS for each genome seperately to make promoter extraction easier later, using GFF_ID to keep the TSS in the same order as the other files
for i in range(0, len(GFFtrainlist)):
    GFFtrainlist[i].to_csv(TRAIN_OUTFILE+str(FILE_ID[i]+".gff"), sep='\t', index=False, header=False)

print('=======================')
print('Done.')
print('Total number of promoters: ' + str(len(pd.concat(GFFlist)) + len(pd.concat(GFFtrainlist))))
print('Number of promoters in Frank: ' + str(len(pd.concat(GFFlist))))
print('Number of promoters for Training: ' + str(len(pd.concat(GFFtrainlist))))
print('=======================')


''' RUNNING CODE - EXTRACTING PROMOTER SEQUENCES FOR TRAINING '''
print('Extracting Promoter Sequences...')

# 1. Access new GFF files to be used to train the model

gff_train_files = []

for ID in FILE_ID:
	gff_train_files.append(TRAIN_OUTFILE + ID + ".gff")

Genome_references = []
for file in FILE_ID:
    try:
        Genome_references.append(remPrevFormatting(FNA_PATH+file+".fna"))
    except:
        Genome_references.append(remPrevFormatting(FNA_PATH+file+".fasta"))

TSS_list = []
for file in gff_train_files:
    temp_TSS = pd.read_csv(file, header=None, sep='\t')
    TSS_list.append(temp_TSS[[0,3,6,8]])

# 2. Extract the promoters from these files (this is why the GFFs for each organism had to be saved seperately)

promoter_list = []
ID = 0
for TSS in TSS_list:
    for i in range(0,len(TSS[3])):
        if TSS[6][i] == '+':
            promoter_list.append(('>TSS|'+ str(TSS[3][i]-PROMOTER_LENGTH) + '-' + str(TSS[3][i]) + '|' + str(PROMOTER_LENGTH) + '|strand+' + '|' + str(TSS[8][i]) + '|' + str(TSS[0][i])))
            promoter_list.append(Genome_references[ID][TSS[3][i]-PROMOTER_LENGTH : TSS[3][i]])
        if TSS[6][i] == '-':
            promoter_list.append(('>TSS|'+ str(TSS[3][i]-1) + '-' + str(TSS[3][i]+PROMOTER_LENGTH-1) + '|' + str(PROMOTER_LENGTH) + '|strand-' + '|' + str(TSS[8][i]) + '|' + str(TSS[0][i])))
            promoter_list.append(reverse_complement(Genome_references[ID][TSS[3][i]-1 : TSS[3][i]+PROMOTER_LENGTH-1]))
    ID += 1


print('done.')
print('=======================')


''' RUNNING CODE - CREATING RANDOM NON-PROMOTERS '''
print('Creating non-promoters...')

# 1. Define how many sequences you want to create (50,000 = amount of bases from a genome like MG1363, as each sequence is 51bp, 50,000*51 = 2,550,000)


# 2. Create the sequences based on genome ATGC content
non_promoter_list = []
total_non_promoters = 0
for i in range(0, len(FILE_ID)):
    NUMBER_OF_NON_PROMOTERS = round((len(pd.read_csv(GFF_PATH+FILE_ID[i]+".gff",sep='\n'))) * NON_PROMOTER_RATIO)
    total_non_promoters += NUMBER_OF_NON_PROMOTERS

    # 2. Create the sequences based on genome ATGC content

    GCcontent = getPercentage(Genome_references[i])
    sequences = makeManyRandoms(NUMBER_OF_NON_PROMOTERS, GCcontent, PROMOTER_LENGTH, promoter_list)
    non_promoter_list.append(sequences)


print('Total number of random non promoters: ' + str(total_non_promoters))
print('done.')
print('=======================')

''' RUNNING CODE - TRAINING THE MODEL '''
print('Training the model...')

# 1. Seperate out a test set
test_proportion = 0.1
print('Using ' +str(test_proportion*100) +' percent of the data for the test set\n')

counter_for_data_split = 100 / (100 * test_proportion)

training_sequences = []
training_response = []
test_sequences = []
test_response = []

list_of_seqs = []
for item in promoter_list:
    if item[0] != '>':
        list_of_seqs.append(item)

for i in range (0,len(list_of_seqs)):
    if i % counter_for_data_split == 0:
        test_sequences.append(list_of_seqs[i])
        test_response.append(1)
    else:
        training_sequences.append(list_of_seqs[i])
        training_response.append(1)

for item in non_promoter_list:
    list_of_seqs = []

    for i in item:
        if i[0] != '>':
            list_of_seqs.append(i)

    for i in range (0,len(list_of_seqs)):
        if i % counter_for_data_split == 0:
            test_sequences.append(list_of_seqs[i])
            test_response.append(0)
        else:
            training_sequences.append(list_of_seqs[i])
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
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
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

''' OUTPUT TEST PREDICTIONS FOR FURTHER INVESTIGATION '''
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


# correct_file_loc = open('data/test_predictions/correct_predictions.txt', 'w')
# false_file_loc = open('data/test_predictions/false_predictions.txt', 'w')

for i in range(0, len(correct_predictions)):
    correct_file_loc.write(str(ID_correct[i]))
    correct_file_loc.write('\n')
    correct_file_loc.write(str(correct_predictions[i]))
    correct_file_loc.write('\n')
correct_file_loc.close()

for i in range(0, len(false_predictions)):
    false_file_loc.write(str(ID_false[i]))
    false_file_loc.write('\n')
    false_file_loc.write(str(false_predictions[i]))
    false_file_loc.write('\n')
false_file_loc.close()

'''How long did the script take'''
print('Time taken: ',datetime.now() - startTime)

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
