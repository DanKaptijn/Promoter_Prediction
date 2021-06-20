'''
Author: Anne de Jong and Daniel Kaptijn
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


https://colab.research.google.com/drive/17E4h5aAOioh5DiTo7MZg4hpL6Z_0FyWr


'''

#import os
import re
import argparse
import pandas as pd
import numpy as np
#from collections import Counter
from statistics import median_high

#from sklearn.externals import joblib
#from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten
#from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

from scipy.cluster.hierarchy import ward, fcluster
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial.distance import pdist
#from scipy.cluster.hierarchy import linkage, dendrogram

from sklearn.metrics import confusion_matrix
#import matplotlib.pyplot as plt
#import itertools


# ---------------------------------------------------------------- parse parameters -------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Promoter Prediction')
parser.add_argument('-sessiondir', dest='sessiondir', help='Session Dir', nargs='?', default='.')
parser.add_argument('-fna', dest='fna', help='Full path and name of FASTA DNA sequence without .fna')
parser.add_argument('-model',dest='model', help='CNN model should be located in the session folder', nargs='?', default='CNN_model.h5')
parser.add_argument('-promlen',dest='PromLen', help='Length of Promoter', nargs='?', default=51)
parser.add_argument('-pval',dest='pvalue', help='p-value cutoff for initial prediction', nargs='?', default=0.99)
parser.add_argument('-out',dest='outPrefix', help='Prefix for Output files', nargs='?', default='ppp')
parser.add_argument('--version', action='version', version='Anne de Jong, version 2.0, Jan 2021')
args = parser.parse_args()


''' Uncomment for local use '''
args.sessiondir = 'G:\\My Drive\\WERK\\PromoterPrediction\\Scripts_Anne\\results'
args.fna = 'G:\My Drive\WERK\PromoterPrediction\Datasets\ManualCurated\Lactococcus_lactis_subsp_cremoris_MG1363_ASM942v1_genomic'
args.model = "CNN_model.h5"
#args.model = "DAN_CNN_model.h5"
args.PromLen  = 51
args.outPrefix = 'ppp'
args.pvalue = 0.99


# ---- be sure PromLen is handled as integer ------
args.PromLen  = int(args.PromLen)

# ----- For clustering -----------------------------
MIN_CLUSTER_SIZE = 1
WINDOW_SIZE = 5
PROBABILITY_CUTOFF = 0.99

# ------ For analysis ------------------------------
TSS = 0 # Denotes how far away from the start of the sequence the TSS is
SHIFT_FOR_NEXT_WINDOW = 1

args.pvalue = float(args.pvalue)




''' ==============================  DEFINING FUNCTIONS =============================================== '''


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



def makeQuerySet(DNA, window_shift):
	query_set = []
	for i in range(0, len(DNA)-args.PromLen,window_shift):
		query_set.append(DNA[i:i+args.PromLen])
	return query_set


def reverse_complement(dna):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ''.join([complement[base] for base in dna[::-1]])

def write_log(S):
	# write to console and logfile
	print(S)
	f = open(args.sessiondir+'/'+"log_makePredictions.log", "a")
	f.write(S + '\n')
	f.close()



write_log('Fasta DNA file ='+args.fna+'.fna')
write_log('GFF file ='+args.fna+'.promoters.gff')
write_log('model file ='+args.model)
write_log('outfile = correct_predictions.txt')
write_log('===========================================================')



''' ==============================  LOAD MODEL ========================================================== '''
write_log('Loading model and formatting DNA for prediction...')

model = load_model(args.sessiondir+'/'+args.model)

DNA_sense = getCleanSeq(args.fna+'.fna')
DNA_antisense = reverse_complement(DNA_sense)
write_log('Number of bases in Clean DNA sequence: '+str(len(DNA_sense)))

test_sense_sequences     = makeQuerySet(DNA_sense,  1)
test_antisense_sequences = makeQuerySet(DNA_antisense,  1)

write_log("Encode sense strand")
input_sense_features = []
for sequence in test_sense_sequences:
	input_sense_features.append(Anne_one_hot_encode(sequence))


write_log("Encode anti-sense strand")
input_antisense_features = []
for sequence in test_antisense_sequences:
	input_antisense_features.append(Anne_one_hot_encode(sequence))


write_log('Sense and anti-sense-feature ready for prediction')
np.set_printoptions(threshold=40)
sense_features     = np.stack(input_sense_features)
antisense_features = np.stack(input_antisense_features)





''' ==============================  MAKING PREDICTIONS ====================================================== '''
write_log('===== Making prediction... ======')

write_log('Create pandas DataFrame as container for promoter data')
promoter_db = pd.DataFrame(columns=['position','score','strand','sequence'])
promoter_db.set_index('position')


# sense strand
write_log('Prediction on sense strand')
predicted_sense_labels = model.predict(sense_features)  # make a numpy.ndarray; original label and predicted label

# save the first 1000 for evaluation
#np.savetxt(args.sessiondir+'/'+"predicted_sense_labels.csv", predicted_sense_labels[0:1000], delimiter="\t")
write_log('\tGet promoters from sense strand with prediction > cutoff')
predicted_sense_promoter_list = []
probabilityValueSense = []
for i in range(0,len(predicted_sense_labels)):
	if (predicted_sense_labels[i][1]) > args.pvalue:
		probabilityValueSense.append(str(predicted_sense_labels[i][1]))
		predicted_sense_promoter_list.append(test_sense_sequences[i])  # Get the DNA sequence
		promoter_db = promoter_db.append({'position': i,'score':predicted_sense_labels[i][1], 'strand' :'+', 'sequence': test_sense_sequences[i]}, ignore_index=True )
write_log('\tNumber of promoters sense strand      : ' + str(len(predicted_sense_promoter_list)))


# anti-sense strand
write_log('Prediction on anti-sense strand')
predicted_antisense_labels = model.predict(antisense_features)  # make a numpy.ndarray; original label and predicted label

write_log('\tGet promoters from anti-sense strand with prediction > cutoff')
predicted_antisense_promoter_list = []
probabilityValueAntisense = []
for i in range(0,len(predicted_antisense_labels)):
	if (predicted_antisense_labels[i][1]) > args.pvalue:
		probabilityValueAntisense.append(str(predicted_antisense_labels[i][1]))
		predicted_antisense_promoter_list.append(test_antisense_sequences[i])   # Get the DNA sequence
		promoter_db = promoter_db.append({'position': i,'score':predicted_antisense_labels[i][1], 'strand' :'-', 'sequence': test_sense_sequences[i]}, ignore_index=True )
write_log('\tNumber of promoters anti-sense strand : ' + str(len(predicted_antisense_promoter_list)))

promoter_db.loc[[0]]
write_log('Total number of predicted promoters: ' + str(len(promoter_db)))
write_log('=======================')

promoter_db.to_csv(args.sessiondir+'/Promoter.db.txt', index = False, sep ='\t')

''' ==============================  MAKING PREDICTIONS ====================================================== '''

# this part is removed because it is slow and will be replaced by promoter_db DataFrame

'''
predicted_promoter_loc = []
for i in range(0,len(DNA_sense)):
    if DNA_sense[i:i+args.PromLen] in predicted_sense_promoter_list:
        loc = [prom for prom in range(0,len(predicted_sense_promoter_list)) if predicted_sense_promoter_list[prom]==DNA_sense[i:i+args.PromLen]]
        temp = ''
        temp += str(i+args.PromLen)
        temp += ("\t+\t")
        temp += str(probabilityValueSense[loc[0]])
        predicted_promoter_loc.append(temp)
    if DNA_antisense[i:i+args.PromLen] in predicted_antisense_promoter_list:
        loc = [prom for prom in range(0,len(predicted_antisense_promoter_list)) if predicted_antisense_promoter_list[prom]==DNA_antisense[i:i+args.PromLen]]
        temp = ''
        temp += str(len(DNA_antisense)-(i+args.PromLen))
        temp += ("\t-\t")
        temp += str(probabilityValueAntisense[loc[0]])
        predicted_promoter_loc.append(temp)

write_log('Number of promoters predicted: '+str(len(predicted_promoter_loc)))
write_log('=======================')

'''



''' ==============================  PERFORM HIERARCHICAL CLUSTERING ========================================= '''
write_log('PERFORM HIERARCHICAL CLUSTERING ')

# 1. Check if clustering is feasible
# This will check if the number of predictions is more than 1% of the total DNA size of the query


new_list = []
if len(promoter_db) < len(DNA_sense) * 0.01:
    write_log('Running hierarchical clustering...')

    # List = []
    # for x in predicted_promoter_loc:
    #    List.append(x.split('\t'))
    # predictedList = pd.DataFrame(List)


    predictedList = promoter_db[['position','strand','score']].copy()

    predicted_sense = predictedList[predictedList['strand']=='+']
    predicted_sense = predicted_sense.apply(pd.to_numeric, errors='ignore')
    predicted_sense[3] = [0]*len(predicted_sense['position']) # Adding a new column of zeroes to make the data appear two dimensional for clustering
    ##### the index is no longer in order due to the split between sense and antisense so I set a new one for ease of use later
    new_index = range(0,len(predicted_sense['position']))
    predicted_sense['index'] = new_index
    predicted_sense = predicted_sense.set_index('index')


    predicted_antisense = predictedList[predictedList['strand']=='-']
    predicted_antisense = predicted_antisense.apply(pd.to_numeric, errors='ignore')
    predicted_antisense = predicted_antisense.iloc[::-1]
    predicted_antisense[3] = [0]*len(predicted_antisense['position'])
    ##### Same as above
    new_index = range(0,len(predicted_antisense['position']))
    predicted_antisense['index'] = new_index
    predicted_antisense = predicted_antisense.set_index('index')

    # write_log(predicted_sense[0].iloc[0:50])

    # plt.scatter((predicted_sense[0].iloc[0:50]), (predicted_sense[3].iloc[0:50]), s=1)
    # plt.show()

    write_log('Hierarchical clustering')
    Xs = predicted_sense[['position',3]]
    Zs = ward(pdist(Xs))
    sense_pred = fclusterdata(Xs, t=WINDOW_SIZE, criterion='distance')


    Xa = predicted_antisense[['position',3]]
    Za = ward(pdist(Xa))
    antisense_pred = fcluster(Za, t=WINDOW_SIZE, criterion='distance')


    sense_dict = {}
    for i in range(0, len(sense_pred)):
        key = int(sense_pred[i])
        value = int(predicted_sense['position'][i])

        if key not in sense_dict.keys():
            sense_dict[key] = [value]
        else:
            new_value = [i for i in sense_dict[key]]
            new_value.append(value)
            sense_dict[key] = new_value


    antisense_dict = {}
    for i in range(0, len(antisense_pred)):
        key = int(antisense_pred[i])
        value = int(predicted_antisense['position'][i])

        if key not in antisense_dict.keys():
            antisense_dict[key] = [value]
        else:
            new_value = [i for i in antisense_dict[key]]
            new_value.append(value)
            antisense_dict[key] = new_value



    for i in sense_dict.keys():
        if len(sense_dict[i]) >= MIN_CLUSTER_SIZE:
            cluster_centre = median_high(sense_dict[i])
            probability = predicted_sense[predicted_sense['position']==cluster_centre]['score'].values[0]
            if probability >= PROBABILITY_CUTOFF:
                temp = ''
                temp += str(cluster_centre)
                temp += '\t+\t'
                temp += str(probability)
                new_list.append(temp)

    for i in antisense_dict.keys():
        if len(antisense_dict[i]) >= MIN_CLUSTER_SIZE:
            cluster_centre = median_high(antisense_dict[i])
            probability = predicted_antisense[predicted_antisense['position']==cluster_centre]['score'].values[0]
            if probability >= PROBABILITY_CUTOFF:
                temp = ''
                temp += str(cluster_centre)
                temp += '\t-\t'
                temp += str(probability)
                new_list.append(temp)


    ''' CONSOLE OUTPUTS '''
    write_log('\n')
    write_log('New number of predictions:      ' +str(int(len(new_list))))
    write_log('Previous number of predictions: ' +str(len(promoter_db)))

	# pd.DataFrame(new_list).to_csv(args.sessiondir+'/Promoter.db.clustered.txt', index = False, sep ='\t')

''' ==============================  ANALYZING THE RESULTS =================================================== '''

''' Still need to be reviewed by Anne '''

write_log('Analyzing the results... by comparing prediction with the original promoters')

# This will use the clustered predictions if clustering was done
if new_list:
    List = []
    for x in new_list:
        List.append(x.split('\t'))
    predictedList = pd.DataFrame(List)

''' TO BE CHECKED: <====================== '''
promotersList = pd.read_csv(args.fna+".promoters.gff", header=None, sep='\t')

predicted_sense = predictedList[predictedList[1]=='+']
predicted_sense = predicted_sense.apply(pd.to_numeric, errors='ignore')
predicted_antisense = predictedList[predictedList[0]=='-']
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

write_log('Total correct (including duplicates):                 '+str(x))
write_log('Number actual correct (no duplicates):                '+str(int(len(predicted_correct)/3)))
write_log('Number missed:                                        '+str(int(len(missed)/2)))
write_log('False positives:                                      '+str(int(len(predicted_wrong)/3)))
write_log('Real total (should be the same as Total):             '+str(len(promotersList[3])))
write_log('Total:                                                '+str(int((len(predicted_TSS)+len(missed))/2)))

''' ==============================  WRITING TO FILE ========================================================= '''
write_log('WRITING TO FILE')

write_to_file = open(args.sessiondir+'/'+args.outPrefix+'/TSS_predicted.txt', "w")
for i in range(0, len(predicted_TSS), 2):
    write_to_file.write(str(predicted_TSS[i]) + '\t' + str(predicted_TSS[i+1]) + '\n')
write_to_file.close()

write_to_file = open(args.sessiondir+'/'+args.outPrefix+'/correct_predictions.txt', "w")
for i in range(0, len(predicted_correct), 3):
        write_to_file.write(str(predicted_correct[i][0]) + '\t' + str(predicted_correct[i+1][0]) + '\t' + str(predicted_correct[i+2][0]) +'\n')
write_to_file.close()

write_to_file = open(args.sessiondir+'/'+args.outPrefix+'/wrong_predictions.txt', "w")
for i in range(0, len(predicted_wrong), 3):
        write_to_file.write(str(predicted_wrong[i]) + '\t' + str(predicted_wrong[i+1][0]) + '\t' + str(predicted_wrong[i+2][0]) + '\n')
write_to_file.close()

write_to_file = open(args.sessiondir+'/'+args.outPrefix+'/missed.txt', "w")
for i in range(0, len(missed), 2):
    write_to_file.write(str(missed[i]) + '\t' + str(missed[i+1][0]) + '\n')
write_to_file.close()


''' ==============================  CONFUSION MATRIX ======================================================== '''
write_log(' CONFUSION MATRIX ')

queryGenome = makeQuerySet(DNA_sense,  SHIFT_FOR_NEXT_WINDOW)
amount_true_negatives = len(queryGenome) - len(promotersList[3])
real_values = ([1]*len(promotersList[3])) + ([0]*amount_true_negatives)
predicted_values = ([1]*int(len(predicted_correct)/3)) + ([0]*int(len(missed)/2)) + ([1]*int(len(predicted_wrong)/3)) + ([0]*int((amount_true_negatives-(len(predicted_wrong)/3))))

cm = confusion_matrix(real_values, predicted_values)
print('\nConfusion matrix:\n',cm)
