#!/usr/bin/env python
# coding=utf-8

""" This script classifies trains a classifier that should be able to tell if a
    word is likely to a portuguese name or not """

import argparse
import random
import numpy as np

from bz2 import BZ2File

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import classification_report

def run(args):
    X_train, y_train, X_test, y_test = split(read_dataset(), test_size=0.99)
    clf = BernoulliNB()

    clf.fit(X_train[:-200], y_train[:-200])
    y_pred = clf.predict(X_train[-200:])

    print(classification_report(y_train[-200:], y_pred))

def split(data, test_size):
    X = TfidfVectorizer(analyzer='char').fit_transform(np.array([d[0] for d in data]))
    y = np.array([d[1] for d in data])

    sss = StratifiedShuffleSplit(y, test_size=test_size, random_state=0)

    for train_index, test_index in sss:
        return (X[train_index], y[train_index],
                X[test_index], y[test_index])

def read_dataset():
    names = [(word, 1) for word in read_words('names.txt.bz2')]
    wikipedia = [(word, 0) for word in read_words('wikipedia-clean.txt.bz2')]

    data = names + wikipedia
    random.shuffle(data)

    print("%d data points were read." % len(data))
    print("%d names and %d not-names" % (len(names), len(wikipedia)))

    return data

def read_words(file_path):
    with BZ2File(file_path) as f:
        return f.read().split('\n')

def parse_args():
    parser = argparse.ArgumentParser(description='')
    #parser.add_argument('path', type=str, help='')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run(args)
