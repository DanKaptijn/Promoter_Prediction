'''
Author: Daniel Kaptijn
date: 24-02-2020
PyVersion: 3.7.6

Aim: Use created model to predict the promoters of other organisms, also to validate using Frank and to do hierarchical clustering if possible
step one:   load a model made by training script
step two:   format genome to be in the correct format for predicting with the CNN model
step three: Predict which sequences are promoters and get back the original sequence format
step four:  Find the location of the last base of each predicted sequence (should be the TSS location)
step five:  Does hierarchical clustering of predictions, if prediction concentration is very high it is not worth it so it will not cluster
step six:   Results analysis, uses GFF file to see how many of the predictions are correct, how many promoters were missed etc.

REQUIREMENTS:
Two input files: fna and GFF file of an organism
'''

print('Starting script...\n')

''' IMPORT MODULES '''
print('Importing modules...')

import os
import re
import pandas as pd
import numpy as np
from collections import Counter
from statistics import median_high

from sklearn.externals import joblib
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

from scipy.cluster.hierarchy import ward, fcluster
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

import time
from datetime import datetime
startTime = datetime.now()

print('Done.')
print('=======================')


''' GLOBAL VARIABLES '''
# ---------------------------------------------------------------- For prediction -------------------------------------------------------------------------------------------
MODEL        = "data/Training/Model/CNN_model.h5"
GENOME_FILE  = "data/Prediction/FNA_files/"
for file in os.listdir(GENOME_FILE):
    result = re.match(".+?\.", file)
    if result:
        GENOME_TO_BE_PREDICTED = GENOME_FILE + result.group(0)
try:
    GENOME_TO_BE_PREDICTED
except:
    print("error: There is no fna file and an fna file is required. Input required file at: %s" %(GENOME_FILE))
    exit()

GFF_FILE = "data/Prediction/GFF_files/"
for file in os.listdir(GFF_FILE):
    result = re.match(".+?\.", file)
    if result:
        CURATED_LIST = GFF_FILE + result.group(0)
try:
    CURATED_LIST
except:
    print("error: There is no GFF file and a GFF file is required. Input required file at: %s" %(GFF_FILE))
    exit()

LENGTH = 51

# ---------------------------------------------------------------- For clustering -------------------------------------------------------------------------------------------

MIN_CLUSTER_SIZE = 1
WINDOW_SIZE = 5
PROBABILITY_CUTOFF = 0.99

# ---------------------------------------------------------------- For analysis -------------------------------------------------------------------------------------------

TSS = 0 # Denotes how far away from the start of the sequence the TSS is
SHIFT_FOR_NEXT_WINDOW = 1

OUTPUT_CORRECT_PREDICTIONS =  'data/Prediction/final_predictions/extra_info/correct_predictions.txt'
OUTPUT_WRONG_PREDICTIONS =    'data/Prediction/final_predictions/extra_info/wrong_predictions.txt'
OUTPUT_PREDICTED_TSS =        'data/Prediction/final_predictions/final_predictions.txt'
OUTPUT_MISSED =               'data/Prediction/final_predictions/extra_info/missed.txt'

print('Genome file ='+GENOME_TO_BE_PREDICTED+"fna")
print('GFF file ='+CURATED_LIST+"gff")
print('model file ='+MODEL)
print('outfile =' +OUTPUT_PREDICTED_TSS)
print('===========================================================')
''' DEFINING FUNCTIONS '''

def Anne_one_hot_encode(seq):
    mapping = dict(zip("ACGT", range(4)))
    seq2 = [mapping[i] for i in seq]
    return np.eye(4)[seq2]


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


def makeQuerySet(DNA, length, window_shift):
	query_set = []
	for i in range(0, len(DNA)-length,window_shift):
		query_set.append(DNA[i:i+length])
	return query_set


def reverse_complement(dna):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ''.join([complement[base] for base in dna[::-1]])


''' RUNNING CODE '''
print('Loading model and formatting genome for prediction...')
model = load_model(MODEL)

try:
    test_sense_sequences        = makeQuerySet(remPrevFormatting(GENOME_TO_BE_PREDICTED+"fna"), LENGTH, 1)
    test_antisense_sequences    = makeQuerySet(reverse_complement(remPrevFormatting(GENOME_TO_BE_PREDICTED+"fna")), LENGTH, 1)
except:
    test_sense_sequences        = makeQuerySet(remPrevFormatting(GENOME_TO_BE_PREDICTED+"fasta"), LENGTH, 1)
    test_antisense_sequences    = makeQuerySet(reverse_complement(remPrevFormatting(GENOME_TO_BE_PREDICTED+"fasta")), LENGTH, 1)
input_sense_features = []
for sequence in test_sense_sequences:
	input_sense_features.append(Anne_one_hot_encode(sequence))

input_antisense_features = []
for sequence in test_antisense_sequences:
	input_antisense_features.append(Anne_one_hot_encode(sequence))


np.set_printoptions(threshold=40)
input_sense_features = np.stack(input_sense_features)
print("Example sequence\n-----------------------")
print('DNA Sequence #1:\n',test_sense_sequences[0][:10],'...',test_sense_sequences[0][-10:])
print('One hot encoding of Sequence #1:\n',input_sense_features[0].T,"\n")

np.set_printoptions(threshold=40)
input_antisense_features = np.stack(input_antisense_features)
print("Example sequence\n-----------------------")
print('DNA Sequence #1:\n',test_antisense_sequences[0][:10],'...',test_antisense_sequences[0][-10:])
print('One hot encoding of Sequence #1:\n',input_antisense_features[0].T,"\n")

sense_features = input_sense_features
antisense_features = input_antisense_features

print('Done.')
print('=======================')


''' MAKING PREDICTIONS '''
print('Making predictions...')

predicted_sense_labels = model.predict(np.stack(sense_features))
predicted_antisense_labels = model.predict(np.stack(antisense_features))

print('Calculating number of predictions...\n')
fing = (round(fing[1]) for fing in predicted_sense_labels)
fing = sum(fing)
fing2 = (round(fing[1]) for fing in predicted_antisense_labels)
fing2 = sum(fing2)
print('Number of Predictions = ' + str(fing+fing2))
print('Interpretting results...\n')

predicted_sense_promoter_list = []
probabilityValueSense = []
for i in range(0,len(predicted_sense_labels)):
    if round(predicted_sense_labels[i][1]) == 1.0:
        probabilityValueSense.append(str(predicted_sense_labels[i][1]))
        promoter = ''
        for j in range(0, len(sense_features[i])):

            if str(sense_features[i][j]) == '[1. 0. 0. 0.]':
                promoter += 'A'

            if str(sense_features[i][j]) == '[0. 1. 0. 0.]':
                promoter += 'C'

            if str(sense_features[i][j]) == '[0. 0. 1. 0.]':
                promoter += 'G'

            if str(sense_features[i][j]) == '[0. 0. 0. 1.]':
                promoter += 'T'
        predicted_sense_promoter_list.append(promoter)

predicted_antisense_promoter_list = []
probabilityValueAntisense = []
for i in range(0,len(predicted_antisense_labels)):
    if round(predicted_antisense_labels[i][1]) == 1.0:
        probabilityValueAntisense.append(str(predicted_antisense_labels[i][1]))
        promoter = ''
        for j in range(0, len(antisense_features[i])):

            if str(antisense_features[i][j]) == '[1. 0. 0. 0.]':
                promoter += 'A'

            if str(antisense_features[i][j]) == '[0. 1. 0. 0.]':
                promoter += 'C'

            if str(antisense_features[i][j]) == '[0. 0. 1. 0.]':
                promoter += 'G'

            if str(antisense_features[i][j]) == '[0. 0. 0. 1.]':
                promoter += 'T'
        predicted_antisense_promoter_list.append(promoter)

print('First 10 predicted promoters: \n',predicted_sense_promoter_list[0:10])
print('First 10 predicted antisense promoters: \n',predicted_antisense_promoter_list[0:10])

print('Mapping predictions...\n')

try:
    full_genome = remPrevFormatting(GENOME_TO_BE_PREDICTED+"fna")
    full_antisense = reverse_complement(remPrevFormatting(GENOME_TO_BE_PREDICTED+"fna"))
except:
    full_genome = remPrevFormatting(GENOME_TO_BE_PREDICTED+"fasta")
    full_antisense = reverse_complement(remPrevFormatting(GENOME_TO_BE_PREDICTED+"fasta"))

predicted_promoter_loc = []
for i in range(0,len(full_genome)):
    if full_genome[i:i+LENGTH] in predicted_sense_promoter_list:
        loc = [prom for prom in range(0,len(predicted_sense_promoter_list)) if predicted_sense_promoter_list[prom]==full_genome[i:i+LENGTH]]
        temp = ''
        temp += str(i+LENGTH)
        temp += ("\t+\t")
        temp += str(probabilityValueSense[loc[0]])
        predicted_promoter_loc.append(temp)
    if full_antisense[i:i+LENGTH] in predicted_antisense_promoter_list:
        loc = [prom for prom in range(0,len(predicted_antisense_promoter_list)) if predicted_antisense_promoter_list[prom]==full_antisense[i:i+LENGTH]]
        temp = ''
        temp += str(len(full_antisense)-(i+LENGTH))
        temp += ("\t-\t")
        temp += str(probabilityValueAntisense[loc[0]])
        predicted_promoter_loc.append(temp)

print('\nNumber of promoters predicted: ',len(predicted_promoter_loc),'\n')
print('=======================')



''' PERFORM HIERARCHICAL CLUSTERING '''
# 1. Check if clustering is feasible
# This will check if the number of predictions is more than 1% of the total genome size of the query
if len(predicted_promoter_loc) > len(full_genome) * 0.01:
    print('=======================')
    print('\nToo many predicted promoters, stacking will not be attempted.\n')
    print('=======================')

    List = []
    for x in predicted_promoter_loc:
        List.append(x.split('\t'))
    predictedList = pd.DataFrame(List)

    time.sleep(10)

if len(predicted_promoter_loc) < len(full_genome) * 0.01:
    print('Running hierarchical clustering...')

    List = []
    for x in predicted_promoter_loc:
        List.append(x.split('\t'))

    predictedList = pd.DataFrame(List)

    predicted_sense = predictedList[predictedList[1]=='+']
    predicted_sense = predicted_sense.apply(pd.to_numeric, errors='ignore')
    predicted_sense[3] = [0]*len(predicted_sense[0]) # Adding a new column of zeroes to make the data appear two dimensional for clustering
    ##### the index is no longer in order due to the split between sense and antisense so I set a new one for ease of use later
    new_index = range(0,len(predicted_sense[0]))
    predicted_sense['index'] = new_index
    predicted_sense = predicted_sense.set_index('index')


    predicted_antisense = predictedList[predictedList[1]=='-']
    predicted_antisense = predicted_antisense.apply(pd.to_numeric, errors='ignore')
    predicted_antisense = predicted_antisense.iloc[::-1]
    predicted_antisense[3] = [0]*len(predicted_antisense[0])
    ##### Same as above
    new_index = range(0,len(predicted_antisense[0]))
    predicted_antisense['index'] = new_index
    predicted_antisense = predicted_antisense.set_index('index')

    # print(predicted_sense[0].iloc[0:50])

    # plt.scatter((predicted_sense[0].iloc[0:50]), (predicted_sense[3].iloc[0:50]), s=1)
    # plt.show()


    Xs = predicted_sense[[0,3]]
    Zs = ward(pdist(Xs))
    sense_pred = fclusterdata(Xs, t=WINDOW_SIZE, criterion='distance')


    Xa = predicted_antisense[[0,3]]
    Za = ward(pdist(Xa))
    antisense_pred = fcluster(Za, t=WINDOW_SIZE, criterion='distance')


    sense_dict = {}
    for i in range(0, len(sense_pred)):
        key = int(sense_pred[i])
        value = int(predicted_sense[0][i])

        if key not in sense_dict.keys():
            sense_dict[key] = [value]
        else:
            new_value = [i for i in sense_dict[key]]
            new_value.append(value)
            sense_dict[key] = new_value


    antisense_dict = {}
    for i in range(0, len(antisense_pred)):
        key = int(antisense_pred[i])
        value = int(predicted_antisense[0][i])

        if key not in antisense_dict.keys():
            antisense_dict[key] = [value]
        else:
            new_value = [i for i in antisense_dict[key]]
            new_value.append(value)
            antisense_dict[key] = new_value


    new_list = []
    for i in sense_dict.keys():
        if len(sense_dict[i]) >= MIN_CLUSTER_SIZE:
            cluster_centre = median_high(sense_dict[i])
            probability = predicted_sense[predicted_sense[0]==cluster_centre][2].values[0]
            if probability >= PROBABILITY_CUTOFF:
                temp = ''
                temp += str(cluster_centre)
                temp += '\t+\t'
                temp += str(probability)
                new_list.append(temp)

    for i in antisense_dict.keys():
        if len(antisense_dict[i]) >= MIN_CLUSTER_SIZE:
            cluster_centre = median_high(antisense_dict[i])
            probability = predicted_antisense[predicted_antisense[0]==cluster_centre][2].values[0]
            if probability >= PROBABILITY_CUTOFF:
                temp = ''
                temp += str(cluster_centre)
                temp += '\t-\t'
                temp += str(probability)
                new_list.append(temp)


    ''' CONSOLE OUTPUTS '''
    print('\n')
    print('New number of predictions:      ', int(len(new_list)))
    print('Previous number of predictions: ', len(predictedList[0]))


''' ANALYZING THE RESULTS '''
print('Analyzing the results...')

# This will use the clustered predictions if clustering was done
if new_list:
    List = []
    for x in new_list:
        List.append(x.split('\t'))
    predictedList = pd.DataFrame(List)

promotersList = pd.read_csv(CURATED_LIST+"gff", header=None, sep='\t')

predicted_sense = predictedList[predictedList[1]=='+']
predicted_sense = predicted_sense.apply(pd.to_numeric, errors='ignore')
predicted_antisense = predictedList[predictedList[1]=='-']
predicted_antisense = predicted_antisense.apply(pd.to_numeric, errors='ignore')

predicted_TSS       = []
predicted_correct   = []
predicted_wrong     = []
missed              = []
x = 0
for i in range(0, len(promotersList[3])):
    temp = []
    y = 0
    for j in range(TSS+promotersList[3][i]-5, TSS+promotersList[3][i]+6):
        if j in predicted_sense[0].values and promotersList[6][i] == '+':
            temp.append((predicted_sense[predicted_sense[0]==j][0].values, predicted_sense[predicted_sense[0]==j][1].values, predicted_sense[predicted_sense[0]==j][2].values))
            y += 1
            x += 1
            if y == 1:
                predicted_TSS.append(promotersList[3][i])
                predicted_TSS.append(promotersList[6][i])
    if len(temp) == 1:
        predicted_correct.append(temp[0][0])
        predicted_correct.append(temp[0][1])
        predicted_correct.append(temp[0][2])
    if(len(temp)) > 1:
        z = temp[0]
        for i in temp:
            if i[2] > z[2]:
                z = i
        predicted_correct.append(z[0])
        predicted_correct.append(z[1])
        predicted_correct.append(z[2])


for i in range(0, len(promotersList[3])):
    temp = []
    y = 0
    for j in range(TSS+promotersList[3][i]-5, TSS+promotersList[3][i]+6):
        if j in predicted_antisense[0].values and promotersList[6][i] == '-':
            temp.append((predicted_antisense[predicted_antisense[0]==j][0].values, predicted_antisense[predicted_antisense[0]==j][1].values, predicted_antisense[predicted_antisense[0]==j][2].values))
            y += 1
            x += 1
            if y == 1:
                predicted_TSS.append(promotersList[3][i])
                predicted_TSS.append(promotersList[6][i])
    if len(temp) == 1:
        predicted_correct.append(temp[0][0])
        predicted_correct.append(temp[0][1])
        predicted_correct.append(temp[0][2])
    if(len(temp)) > 1:
        z = temp[0]
        for i in temp:
            if i[2] > z[2]:
                z = i
        predicted_correct.append(z[0])
        predicted_correct.append(z[1])
        predicted_correct.append(z[2])


check_list = []
for i in range(0, len(predicted_correct), 3):
    check_list.append(predicted_correct[i][0])

for i in predictedList[0]:
    if int(i) not in check_list:
        predicted_wrong.append(i)
        predicted_wrong.append(predictedList[predictedList[0]==i][1].values)
        predicted_wrong.append(predictedList[predictedList[0]==i][2].values)

check_list = []
for i in range(0, len(predicted_TSS), 2):
    check_list.append(predicted_TSS[i])

for i in promotersList[3].values:
    if int(i) not in check_list:
        missed.append(i)
        missed.append(promotersList[promotersList[3]==i][6].values)

print('Total correct (including duplicates):                 ',x)
print('Number actual correct (no duplicates):                ',int(len(predicted_correct)/3))
print('Number missed:                                        ',int(len(missed)/2))
print('False positives:                                      ',int(len(predicted_wrong)/3))
print('Real total (should be the same as Total):             ',len(promotersList[3]))
print('Total:                                                ',int((len(predicted_TSS)+len(missed))/2))


'''WRITING TO FILE'''
write_to_file = open(OUTPUT_PREDICTED_TSS, "w")
for i in range(0, len(predicted_TSS), 2):
    write_to_file.write(str(predicted_TSS[i]) + '\t' + str(predicted_TSS[i+1]) + '\n')
write_to_file.close()

write_to_file = open(OUTPUT_CORRECT_PREDICTIONS, "w")
for i in range(0, len(predicted_correct), 3):
        write_to_file.write(str(predicted_correct[i][0]) + '\t' + str(predicted_correct[i+1][0]) + '\t' + str(predicted_correct[i+2][0]) +'\n')
write_to_file.close()

write_to_file = open(OUTPUT_WRONG_PREDICTIONS, "w")
for i in range(0, len(predicted_wrong), 3):
        write_to_file.write(str(predicted_wrong[i]) + '\t' + str(predicted_wrong[i+1][0]) + '\t' + str(predicted_wrong[i+2][0]) + '\n')
write_to_file.close()

write_to_file = open(OUTPUT_MISSED, "w")
for i in range(0, len(missed), 2):
    write_to_file.write(str(missed[i]) + '\t' + str(missed[i+1][0]) + '\n')
write_to_file.close()


''' CONFUSION MATRIX '''
queryGenome = makeQuerySet(full_genome, LENGTH, SHIFT_FOR_NEXT_WINDOW)
amount_true_negatives = len(queryGenome) - len(promotersList[3])
real_values = ([1]*len(promotersList[3])) + ([0]*amount_true_negatives)
predicted_values = ([1]*int(len(predicted_correct)/3)) + ([0]*int(len(missed)/2)) + ([1]*int(len(predicted_wrong)/3)) + ([0]*int((amount_true_negatives-(len(predicted_wrong)/3))))


cm = confusion_matrix(real_values,
                      predicted_values)
print('\nConfusion matrix:\n',cm)


print('Time taken: ',datetime.now() - startTime)
