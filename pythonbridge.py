import gensim as gnsm
import array
import numpy as np
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("nmbr", help="Number of child articles",
                    type=int)

args = parser.parse_args()
ndat = args.nmbr  + 1
arr1 = []
arr2 = []
sim  = 0
model = gnsm.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

with open("ababa.dat","r") as f:
    for line in f:
        for word in line.split():
           arr1.append(word)

for i in xrange(ndat):
    with open('Data_{0}.dat'.format(i),'r') as f:
        for line in f:
            for word in line.split():
                arr2.append(word)

#    if len(arr1)>len(arr2):
    la1 = len(arr1)
    la2 = len(arr2)
    vals = [0]*len(arr2)*len(arr1)
    maxv = [0]*len(arr2)*len(arr1)

#    else:
#        la1 = len(arr1)
#        la2 = len(arr2)
#        vals = [0]*len(arr2)*len(arr1)
#        maxv = [0]*len(arr2)*len(arr1)

    for ii in range(la1):
        for jj in range(la2):
            try:
#                    print(len(vals),jj)
                    vals[jj] = model.similarity(arr1[ii],arr2[jj])
                #sim =model.n_similarity(arr2,arr1)
            except KeyError:
                sim = sim +1
        #        print(sim)

        maxv[ii] = np.max(vals)
    asd = np.sum(maxv)/(len(arr1)-sim)
    print json.dumps(asd, indent = 4)

#model.similarity('physics','living')
