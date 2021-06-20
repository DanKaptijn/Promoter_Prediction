'''
Author: Daniel Kaptijn
date: 26-01-2021
PyVersion: 3.7.6

Aim: To find out the average and range of ATCG contents for all the promoters in an organism.

REQUIREMENTS:
Two input files: fna and GFF file of an organism
'''

''' IMPORT MODULES '''
import os
import re
import pandas as pd
from statistics import median


''' GLOBAL VARIABLES '''
FILE_ID_GFF = ''
FILE_ID_FNA = ''
GFF_PATH = "data/Training/GFF_files/"
FNA_PATH = "data/Training/FNA_files/"
for file in os.listdir(FNA_PATH):
    result = re.match(".+?(?=\.)", file)
    if result:
        FILE_ID_FNA += result.group(0)

for file in os.listdir(GFF_PATH):
    result = re.match(".+?(?=\.)", file)
    if result:
        FILE_ID_GFF += result.group(0)

GFF = GFF_PATH + FILE_ID_GFF + '.gff'
FNA = FNA_PATH + FILE_ID_FNA

print('gff file =' + GFF)
print('fna file = ' + FNA + '.fna\n')

PROMOTER_LENGTH = 51



''' DEFINE FUNCTIONS '''

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


def getPercentage(genome):
	counter = 0
	Acounter = 0
	Tcounter = 0
	for i in genome:
	    counter += 1
	    if i == 'A':
	        Acounter += 1
	    if i == 'T':
	        Tcounter += 1
	percentageAT = round(((Acounter+Tcounter)/counter)*100,2)
	return(percentageAT)


def Mean(lst):
    return(round(sum(lst) / len(lst),2))



''' RUNNING CODE - EXTRACTING PROMOTERS '''

Genome_references = []
try:
    Genome_references.append(remPrevFormatting(FNA+".fna"))
except:
    Genome_references.append(remPrevFormatting(FNA+".fasta"))

TSS_list = []
temp_TSS = pd.read_csv(GFF, header=None, sep='\t')
TSS_list.append(temp_TSS[[0,3,6,8]])


promoter_list = []
ID = 0
for TSS in TSS_list:
    for i in range(0,len(TSS[3])):
        if TSS[6][i] == '+':
            promoter_list.append(Genome_references[ID][TSS[3][i]-PROMOTER_LENGTH : TSS[3][i]])
        if TSS[6][i] == '-':
            promoter_list.append(reverse_complement(Genome_references[ID][TSS[3][i]-1 : TSS[3][i]+PROMOTER_LENGTH-1]))
    ID += 1

print('First promoter sequence = ' + str(promoter_list[0]))
ATcontent = []
for promoter in promoter_list:
    ATcontent.append(getPercentage(promoter))

ATcontent.sort()

print('Least AT content in a promoter = ' + str(min(ATcontent)) + '%')
print('Most AT content in a promoter = ' + str(max(ATcontent)) + '%')
print('Average AT content = ' + str(Mean(ATcontent)) + '%')
print('Median AT content = ' + str(median(ATcontent)) + '%')
