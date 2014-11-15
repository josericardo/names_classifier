#!/usr/bin/env python
# coding=utf-8

""" This script classifies trains a classifier that should be able to tell if a
    word is likely to a portuguese name or not """

import argparse
import random
import numpy as np
import os

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn import cross_validation

def run(args):
    X_train, y_train = read_dataset('train')
    clf = BernoulliNB()

    scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=10,
                                              n_jobs=4, scoring='f1', verbose=1)

    print("F1-Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def read_dataset(basedir):
    names = [(word, 1) for word in read_words(os.path.join(basedir, 'names.txt'))]
    wikipedia = [(word, 0) for word in read_words(os.path.join(basedir, 'words.txt'))]

    data = names + wikipedia
    random.shuffle(data)

    print("%d data points were read." % len(data))
    print("%d names and %d not-names" % (len(names), len(wikipedia)))

    X = TfidfVectorizer(analyzer='char', ngram_range=(2,3)).fit_transform(np.array([d[0] for d in data]))
    y = np.array([d[1] for d in data])

    return X, y

def read_words(file_path):
    with open(file_path) as f:
        return f.read().split('\n')

def parse_args():
    parser = argparse.ArgumentParser(description='')
    #parser.add_argument('path', type=str, help='')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run(args)
