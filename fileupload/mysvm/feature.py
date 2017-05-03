# feature.py
# Author: Indrajith Indraprstham
# Date: Sun Apr 28 2017
# Modified on : Tue May  2 16:50:36 IST 2017

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

def extract(file):
    (rate, data) = scipy.io.wavfile.read(file)
    mfcc_feat = mfcc(data,rate)
    #redusing mfcc dimension to 104
    mm = np.transpose(mfcc_feat)
    mf = np.mean(mm,axis=1)
    cf = np.cov(mm)
    ff=mf  

    #ff is a vector of size 104
    for i in range(mm.shape[0]):
        ff = np.append(ff,np.diag(cf,i))

    return ff


def extract_all(audio_dir):
    all_music_files = glob.glob(audio_dir)
    all_music_files.sort()
    all_mfcc = np.array([])
    loop_count = 0
    flag = True

    for file_name in all_music_files:
        (rate, data) = scipy.io.wavfile.read(file_name)
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
        loop_count += 1

    return all_mfcc


def extract_ratio(train_ratio, test_ratio, audio_dir):
    """
    Extract audio in a ratio from the directory for training and testing 
    returns two numpy array Xtrain and Xtest 
    audio_dir: should be of the for : "wav/*/*.wav"
    """
    if test_ratio + train_ratio != 1:
        print("ratios should add up to 1\n")
        return

    all_music_files = glob.glob(audio_dir)
    all_music_files.sort()
    all_mfcc = np.array([])
    flag = True

    Testing = False
    Training = True

    #initialize the array
    all_mfcc = np.array([])

    count = 0; #training
    #count = test_ratio; 
    loop_count = -1
    flag = True
    

    for train_test_loop in range(2):      
        #extract mfcc features for the audio files
        for file_name in all_music_files: 
            #for training select only train_ratio songs from each
            if Training:
                loop_count += 1
                if loop_count % 100 == 0:
                    count = 0
                if count == train_ratio * 100:
                    continue    #selects only train_ratio songs in every 100 songs
                count += 1
            
            #for testing select last test_ratio songs from each genre
            if Testing:
                loop_count += 1
                if (loop_count + (test_ratio * 100)) % 100 == 0 and loop_count:
                    count = 0
                    print('--'*10)
            
                if count == test_ratio * 100:
                    continue
                count += 1

            if Training or Testing:
                (rate, data) = scipy.io.wavfile.read(file_name)
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

        if train_test_loop == 0:
            print('\n'*10,'====Collected Training data===','\n'*20)
            print('\n','====Collecting Testing data===','\n')
            Xtrain = all_mfcc
            count = test_ratio * 100
            Testing = True
            Training = False
            loop_count = -1
            all_mfcc = np.array([])
            flag = True
            print("Xtrain.shape:", Xtrain.shape)

        if train_test_loop == 1:
            print('\n','====Collected Testing data===','\n')
            Xtest = all_mfcc
            print("Xtest.shape:", Xtest.shape)

    return Xtrain, Xtest

def geny(n):
    """
    Generate Y for the dataset
    """
    y = [np.ones(n),np.ones(n)*2,np.ones(n)*3,np.ones(n)*4,np.ones(n)*5, \
    np.ones(n)*6,np.ones(n)*7,np.ones(n)*8,np.ones(n)*9,np.ones(n)*10]

    return np.array(flatten(y))

def gen_suby(sub,n):
    """
    generates y for a subclass 
    usage: gen_suby(sub,n)
    """
    y = np.array([])
    flag = True
    for i in range(1, sub + 1):
        if flag:
            y = np.ones(n) * i
            flag = False
        else:
            y = np.vstack([y,np.ones(n) * i])
    return np.array(flatten(y))
