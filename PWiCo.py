'''
Author: Daniel Kaptijn
Date: 18/10/2019
PyVersion: 3.7.3

Aim: Create Own Pattern Recognition Software
'''


'''
THOUGHTS
-Finding one pre-defined pattern should be easy
-Recognising previously unknown patterns will be hard
-Midway point could be to find pre-defined pattern with gaps
-Create random strings as sequences to test the functions
-unsupervised will require finding a pattern multiple times
--longer sequence would therefore be preferable
--pattern would necessarily be max 0.5 of full sequence depending on number of times to be found
'''


'''IMPORT MODULES'''

import sys
import math


'''GLOBAL VARIABLES'''
PATTERN = "ATGATATATAT"
SEQUENCE = "GTATCTCAGCATCGCATCATATCAGCATCGACTAGCGATATATCATACTACTACAGCTACTAGCATATAGATAATGATAGA"
NO_PAT_SEQ = "PORT"
ALT_SEQ = "FIREBEE"


'''DEFINING FUNCTIONS'''

# This should recognise if the pattern is present, where it is and how many times it occurs
def simplePatRec(pat, seq):
    if pat in seq:
        amount = 0
        for i in range(len(seq)):
            if seq[i:i+4] == pat:
                amount += 1

    if pat in seq:
        print("\nThe pattern \"%s\" is contained within the input sequence %d times, and it is located at:" % (pat,amount))
    else:
        print("\nThe pattern \"%s\" is not contained within the input sequence\n" % (pat))

    for i in range(len(seq)):
        if seq[i:i+4] == pat:
            print("letters %d to %d" % (i+1,i+4))

# Similar to the simple but should also include partial similarities, limit similarity to 0.50 for some ease
def detailedPatRec(pat, seq):
    check = False
    if pat in seq:
        check = True
        simplePatRec(pat, seq)

    if check == False:
        print("Query sequence is not contained fully in comparison sequence.\n")
        length = math.ceil(len(pat)/2)
        for i in range(length+1):
            pat1 = pat[0:len(pat)-i]
            pat2 = pat[i:len(pat)]
            score = (len(pat1) / len(pat)) * 100
            if pat1 in seq:
                check = True
                print("\nThe longest section of query found was \"%s\" With a similarity score of %d" % (pat1,score))
                if input("type \'yes\' for locations: ") == "yes":
                    for i in range(len(seq)):
                        if seq[i:i+len(pat1)] == pat1:
                            print("Found at:\nletters %d to %d" % (i+1,i+len(pat1)))
            if pat2 in seq:
                check = True
                print("\nThe longest section of query found was \"%s\" With a similarity score of %d" % (pat2,score))
                if input("type \'yes\' for locations: ") == "yes":
                    for i in range(len(seq)):
                        if seq[i:i+len(pat2)] == pat2:
                            print("letters %d to %d" % (i+1,i+len(pat2)))
            if check == True:
                exit()

        print("Less than 50% similar and so search has ended")


def unsupervisedPatRec(seq):
    print("Not yet available")


'''RUNNING CODE'''

if len(sys.argv) == 1:
    print("\nERROR: This script requires input:\n\
    BASIC: type basic for a simple whole match of your query\n\
    MED: type med for a more detailed search of your query\n\
    COMP: type comp to run an unsupervised pattern recognition")

if len(sys.argv) > 1:
    if sys.argv[1] == "basic":
        answer = simplePatRec(PATTERN, SEQUENCE)
    elif sys.argv[1] == "med":
        answer = detailedPatRec(PATTERN, SEQUENCE)
    elif sys.argv[1] == "comp":
        answer = unsupervisedPatRec(SEQUENCE)
    elif sys.argv[1] == "test":
        # Should test each function when no similarity present (not possible for unsupervised)
        print("\n*Testing for no recognition:*")
        nulltest = simplePatRec(PATTERN, NO_PAT_SEQ)
    else:
        print("\nERROR: This script requires specific input:\n\
    BASIC: type basic for a simple whole match of your query\n\
    MED: type med for a more detailed search of your query\n\
    COMP: type comp to run an unsupervised pattern recognition")
