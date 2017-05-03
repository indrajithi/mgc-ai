# File: svm.py
# Author: Indrajith Indraptrastham
# Date: Sun Apr 28 2017

import numpy as np
import scipy.io.wavfile
from python_speech_features import mfcc
import glob
import collections

def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

#get the relative path of the files
a = glob.glob("wav/*/*.wav")
Testing = False
Training = True
a.sort()
#initialize the array
all_mfcc = np.array([])

count = 0; #training
#count = 30; #test
loop_count = -1
flag = True
#extract mfcc features for the audio files
for i in a: 
    
    #for training select only 90 songs from each
    if Training:
        loop_count += 1
        if loop_count % 100 == 0:
            count = 0
        if count == 70:
            continue    #selects only 90 songs in every 100 songs
        count += 1
    
    #for testing select last 10 songs from each genre
    if Testing:
        loop_count += 1
        if (loop_count + 30) % 100 == 0 and loop_count:
            count = 0
            print('--'*10)
    
        if count == 30:
            continue
        count += 1


    (rate, data) = scipy.io.wavfile.read(i)
    mfcc_feat = mfcc(data,rate)
    #redusing mfcc dimension to 104
    mm = np.transpose(mfcc_feat)
    mf = np.mean(mm,axis=1)
    cf = np.cov(mm)
    ff=mf  

    #ff is a vector of size 104
    for i in range(mm.shape[0]):
        ff = np.append(ff,np.diag(cf,i))

    #re initializing to size 104
    if flag:
        all_mfcc = ff;
        print('*'*20)
        flag = False      
    else:
        all_mfcc = np.vstack([all_mfcc,ff])
    
    print("loooping----",loop_count)
    print("all_mfcc.shape:",all_mfcc.shape)

y=[np.ones(70),np.ones(70)*2,np.ones(70)*3,np.ones(70)*4,np.ones(70)*5, \
np.ones(70)*6,np.ones(70)*7,np.ones(70)*8,np.ones(70)*9,np.ones(70)*10]

yt=[np.ones(30),np.ones(30)*2,np.ones(30)*3,np.ones(30)*4,np.ones(30)*5, \
np.ones(30)*6,np.ones(30)*7,np.ones(30)*8,np.ones(30)*9,np.ones(30)*10]

y = flatten(y)
yt = flatten(yt)



