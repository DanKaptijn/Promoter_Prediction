'''
Author: Dan Kaptijn
Date: 13/05/2020
PyVersion: 3.7.6

Aim: This script will output an excel document which consists of various information
about each predicted promoter, this information include:
-Promoter Name
-Is it intergenic
-Is it the primary promoter (first promoter upstream of a gene)
-The 5' UTR size
-Does it have a -10 motif
-Does it have a -35 motif

Aim 2: Output a list of genes that do not have a promoter predicted upstream
'''
print('Starting script and importing modules...\n')


'''IMPORT MODULES'''
import os
import re
import pandas as pd
import numpy as np

from datetime import datetime
startTime = datetime.now()

'''INPUTS AND OUTPUTS'''
GENOME_FILE = "data/Further/FNA_file/"
for file in os.listdir(GENOME_FILE):
    GENOME_TO_BE_PREDICTED = GENOME_FILE + file
    result = re.match(".+?(?=\.)", file)
try:
    GENOME_TO_BE_PREDICTED
except:
    print("error: There is no fna file and an fna file is required. Input required file at: %s" %(GENOME_FILE))
    exit()

ANNOTATION_FILE = "data/Further/Annotation_file/"
for file in os.listdir(ANNOTATION_FILE):
    GENOME_ANNOTATIONS = ANNOTATION_FILE + file
try:
    ANNOTATION_FILE
except:
    print("error: There is no GFF annotations file which is required. Input required file at: %s" %(ANNOTATION_FILE))
    exit()

PREDDICTED_PROMOTERS_FILE = "data/Further/Predicted Promoters/"
for file in os.listdir(PREDDICTED_PROMOTERS_FILE):
    PREDICTED_PROMOTERS = PREDDICTED_PROMOTERS_FILE + file
try:
    PREDICTED_PROMOTERS
except:
    print("error: There is no file of predicted promoters which is required. Input required file at: %s" %(PREDDICTED_PROMOTERS_FILE))
    exit()

OUTPUT_PROMOTER_FILE = "data/Further/Output/" + "Results_" + result.group(0) + ".xlsx"
OUTPUT_GENE_LIST  = "data/Further/Output/" + "Gene_List_" + result.group(0) + ".txt"

print("Genome file: " + GENOME_TO_BE_PREDICTED)
print("Predicted Promoters file: " + PREDICTED_PROMOTERS)
print("GFF annotation file: " + GENOME_ANNOTATIONS)
print("Outfile Promoters: " + OUTPUT_PROMOTER_FILE)
print("Outfile Gene List: " + OUTPUT_GENE_LIST)
print('===========================================================')


''' GLOBAL VARIABLES '''
INTERGENIC_REGION = 200
PROMOTER_LENGTH = 51


'''DEFININF FUNCTIONS'''
print("Defining functions...\n")

def intergenic(strand_promoter, loc_promoter, list_genes):
    out = 'NA'
    if strand_promoter == '+':
        for i in range(loc_promoter, loc_promoter+1000):
            if i in list(list_genes[4]):
                temp = list_genes[list_genes[4]==i]
                if temp[6][temp.index[0]] == strand_promoter:
                    out = 'no'
                    break
            if i in list(list_genes[3]):
                temp = list_genes[list_genes[3]==i]
                if temp[6][temp.index[0]] == strand_promoter:
                    out = 'yes'
                    break
    if strand_promoter == '-':
        for i in range(loc_promoter, loc_promoter-1000, -1):
            if i in list(list_genes[3]):
                temp = list_genes[list_genes[3]==i]
                if temp[6][temp.index[0]] == strand_promoter:
                    out = 'no'
                    break
            if i in list(list_genes[4]):
                temp = list_genes[list_genes[4]==i]
                if temp[6][temp.index[0]] == strand_promoter:
                    out = 'yes'
                    break

    return out


def primary(intergenic, strand_promoter, loc_promoter, list_promoters, list_genes):
    out = 'no'
    if intergenic == 'yes':
        if strand_promoter == '+':
            for i in range(loc_promoter, loc_promoter+INTERGENIC_REGION):
                if i in list(list_genes[3]):
                    temp = list_genes[list_genes[3]==i]
                    if temp[6][temp.index[0]] == strand_promoter:
                        out = 'yes'
                        break
                if i+1 in list(list_promoters[0]):
                    temp = list_promoters[list_promoters[0]==i+1]
                    if temp[1][temp.index[0]] == strand_promoter:
                        out = 'no'
                        break
        if strand_promoter == '-':
            for i in range(loc_promoter, loc_promoter-INTERGENIC_REGION, -1):
                if i in list(list_genes[4]):
                    temp = list_genes[list_genes[4]==i]
                    if temp[6][temp.index[0]] == strand_promoter:
                        out = 'yes'
                        break
                if i-1 in list(list_promoters[0]):
                    temp = list_promoters[list_promoters[0]==i-1]
                    if temp[1][temp.index[0]] == strand_promoter:
                        out = 'no'
                        break

    return out


def five_UTR_distance(intergenic, strand_promoter, loc_promoter, list_genes):
    search = True
    dist_to_gene = 'NA'
    if intergenic == 'yes':
        if strand_promoter == '+':
            i = loc_promoter
            while search:
                if i in list(list_genes[3]):
                    temp = list_genes[list_genes[3]==i]
                    if temp[6][temp.index[0]] == strand_promoter:
                        dist_to_gene = i - loc_promoter
                        search = False
                i += 1
        if strand_promoter == '-':
            i = loc_promoter
            while search:
                if i in list(list_genes[4]):
                    temp = list_genes[list_genes[4]==i]
                    if temp[6][temp.index[0]] == strand_promoter:
                        dist_to_gene = loc_promoter - i
                        search = False
                i -= 1
                if i == 0:
                    dist_to_gene = 'NA'
                    search = False
    return dist_to_gene


def reverse_complement(dna):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ''.join([complement[base] for base in dna[::-1]])

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


def motifs(motif, seq):
    Total = 0
    if motif == 'TATAAT':
        if seq[0] == 'T':
            Total += 1
        if seq[1] == 'A':
            Total += 1
        if seq[2] == 'T':
            Total += 1
        if seq[3] == 'A':
            Total += 1
        if seq[4] == 'A':
            Total += 1
        if seq[5] == 'T':
            Total += 1

    if motif == 'TTGACA':
        if seq[0] == 'T':
            Total += 1
        if seq[1] == 'T':
            Total += 1
        if seq[2] == 'G':
            Total += 1
        if seq[3] == 'A':
            Total += 1
        if seq[4] == 'C':
            Total += 1
        if seq[5] == 'A':
            Total += 1

    if Total > 3:
        return True
    if Total < 4:
        return False


def extractPromoterSequence(genome, len_promoter, strand_promoter, loc_promoter):
    ten = 'no'
    thirtyfive = 'no'
    if strand_promoter == '+':
        sequence = genome[loc_promoter-len_promoter+1:loc_promoter+1]
        for i in range(51-17, 51-6):
            n = motifs('TATAAT', sequence[i:i+6])
            if n:
                ten = 'yes'
                break
        for i in range(0, 20):
            n = motifs('TTGACA', sequence[i:i+6])
            if n:
                thirtyfive = 'yes'
                break
    if strand_promoter == '-':
        sequence = reverse_complement(genome[loc_promoter:loc_promoter+len_promoter])
        for i in range(51-17, 51-6):
            n = motifs('TATAAT', sequence[i:i+6])
            if n:
                ten = 'yes'
                break
        for i in range(0, 20):
            n = motifs('TTGACA', sequence[i:i+6])
            if n:
                thirtyfive = 'yes'
                break

    return ten, thirtyfive



'''RUNNING CODE'''
print("Creating list of genes which have no intergenic predicted promoters...\n")

# Creating two dataframes of genes to compare with the predictions in order to see
# which genes do not have a promoter within 200bp of its start
genes = pd.read_csv(GENOME_ANNOTATIONS, header=None, comment='#', sep='\t')
genes = genes[genes[2]=='gene']
new_index = range(0,len(genes[0]))
genes['index'] = new_index
genes = genes.set_index('index')

print("Number of Genes: " + str(len(genes)) )
genes_sense = genes[genes[6]=='+']
print(genes_sense.head())
print("Number of sense Genes: " + str(len(genes_sense)) )
genes_antisense = genes[genes[6]=='-']
print(genes_antisense.head())
print("Number of antisense Genes: " + str(len(genes_antisense)) + '\n')

# Creating two dataframes of predictions to compare with the gene dataframes
pred_promoters = pd.read_csv(PREDICTED_PROMOTERS, header=None, comment='#', sep='\t')
print("Number of predicted promoters: " + str(len(pred_promoters)) )
pred_promoters_sense = pred_promoters[pred_promoters[1]=='+']
print(pred_promoters_sense.head())
print("Number of predicted sense promoters: " + str(len(pred_promoters_sense)) )
pred_promoters_antisense = pred_promoters[pred_promoters[1]=='-']
print(pred_promoters_antisense.head())
print("Number of predicted antisense promoters: " + str(len(pred_promoters_antisense)) )

pred_sense_numbers = []
for i in pred_promoters_sense[0]:
    pred_sense_numbers.append(i)

pred_antisense_numbers = []
for i in pred_promoters_antisense[0]:
    pred_antisense_numbers.append(i)

gene_list = []
for i in genes_sense[3]:
    counter = 0
    for j in range(i-INTERGENIC_REGION, i+1):
        counter += 1
        if j in pred_sense_numbers:
            break
    if counter > INTERGENIC_REGION:
        temp = genes_sense[genes_sense[3]==i]
        temp = temp[8].str.split(';')
        gene_list.append(temp[temp.index[0]][1])

for i in genes_antisense[4]:
    counter = 0
    for j in range(i, i+INTERGENIC_REGION):
        counter += 1
        if j in pred_antisense_numbers:
            break
    if counter > INTERGENIC_REGION:
        temp = genes_antisense[genes_antisense[3]==i]
        temp = temp[8].str.split(';')
        gene_list.append(temp[temp.index[0]][1])

write_to_file = open(OUTPUT_GENE_LIST, 'w')
for i in gene_list:
    write_to_file.write(str(i)[5:]+'\n')
write_to_file.close()

print("Done.")
print("Number of genes that do not have an intergenic promoter predicted: " + str(len(gene_list)) )
print("List of genes without intergenic promoter: " + str(gene_list[0:10]) )
print('===========================================================')


print("\nCreating excel file of promoter properties...\n")

# STEP ONE: Creating a sorted list of all the predicted promoters so that the names will be in the correct order
pred_promoters = pred_promoters.sort_values(by=0)
new_index = range(0,len(pred_promoters[0]))
pred_promoters['index'] = new_index
pred_promoters = pred_promoters.set_index('index')
print(pred_promoters[0:5])
print('\n')

# STEP TWO: make an empty dataframe to store the information
promoter_info = {'Promoter':[],'Intergenic':[],'Primary':[],'5\' UTR size':[],'-10':[],'-35':[]}

# STEP THREE: a for loop needs to be used to go through each promoter at a time and add each attribute to a file
genome = remPrevFormatting(GENOME_TO_BE_PREDICTED)
print('Collecting information about each promoter...')
print('0% complete', end='\r')
for i in range(0, len(pred_promoters[0])):

    for j in range(1,101):
        if i == round(len(pred_promoters[0]) * (j/100)):
            print(str(j)+'% complete', end='\r')
            break

    promoter_info['Promoter'].append('Promoter '+str(i+1))

    result_intergenic = intergenic(pred_promoters[1][i], pred_promoters[0][i], genes)
    promoter_info['Intergenic'].append(result_intergenic)

    result = primary(result_intergenic, pred_promoters[1][i], pred_promoters[0][i], pred_promoters, genes)
    promoter_info['Primary'].append(result)

    result = five_UTR_distance(result_intergenic, pred_promoters[1][i], pred_promoters[0][i], genes)
    promoter_info['5\' UTR size'].append(result)

    r10,r35 = extractPromoterSequence(genome, PROMOTER_LENGTH, pred_promoters[1][i], pred_promoters[0][i])
    promoter_info['-10'].append(r10)
    promoter_info['-35'].append(r35)

# STEP FOUR: make a dataframe and save it as an excel file
print('100% complete\n')
print("Writing to file...")
promoter_df = pd.DataFrame(promoter_info, columns=['Promoter','Intergenic','Primary','5\' UTR size','-10','-35'])
promoter_df.to_excel(OUTPUT_PROMOTER_FILE, index = False)

print("Done.")
print("Head and tail of promoter information: " + str(promoter_df.head()) +str(promoter_df.tail()) )
print('===========================================================')

'''How long did the script take'''
print('Time taken: ',datetime.now() - startTime)
