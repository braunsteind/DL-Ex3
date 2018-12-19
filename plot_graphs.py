# -*- coding: utf-8 -*-

STUDENT = {'name': 'Coral Malachi_Daniel Braunstein',
           'ID': '314882853_312510167'}
"""
    in this file we'll plot 2 graphs
    Graph 1 is the learning curves for the POS data (the dev-set accuracies).
    It should have 4 lines, corresponding to input representations (a), (b), (c), (d)
    above. 
     Graph 2 is the learning curves for the NER data, again with 4 lines.
"""
import cPickle as pickle
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import collections as co

def load_accuracy_list(pkl_name):
    """

    :param pkl_name:
    :return: the function load the list of accuracies from pkl file
    """
    with open(pkl_name) as dicts_file:
        dict = pickle.load(dicts_file)
    return co.OrderedDict(sorted(dict.items()))

a_ner = load_accuracy_list("a_model_ner_changes.pkl")
b_ner = load_accuracy_list("b_model_ner_changes.pkl")
c_ner = load_accuracy_list("c_model_ner_changes.pkl")
d_ner = load_accuracy_list("d_model_ner_changes.pkl")

a_pos = load_accuracy_list("a_model_pos_fixed.pkl")
b_pos = load_accuracy_list("b_model_pos_fixed.pkl")
c_pos = load_accuracy_list("c_model_pos_fixed.pkl")
d_pos = load_accuracy_list("d_model_pos_fixed.pkl")

label1, = plt.plot(a_pos.keys(), a_pos.values(), "k-", label='a(pos)')
label2, = plt.plot(b_pos.keys(), b_pos.values(), "g-", label='b(pos)')
label3, = plt.plot(c_pos.keys(), c_pos.values(), "r-", label='c(pos)')
label4, = plt.plot(d_pos.keys(), d_pos.values(), "b-", label='d(pos)')
plt.legend(handler_map={label1: HandlerLine2D(numpoints=4)})
plt.ylabel("the accuracy")
plt.xlabel("iter number / 100")
plt.show()

label1, = plt.plot(a_ner.keys(), a_ner.values(), "k-", label='a(ner)')
label2, = plt.plot(b_ner.keys(), b_ner.values(), "g-", label='b(ner)')
label3, = plt.plot(c_ner.keys(), c_ner.values(), "r-", label='c(ner)')
label4, = plt.plot(d_ner.keys(), d_ner.values(), "b-", label='d(ner)')

plt.legend(handler_map={label1: HandlerLine2D(numpoints=4)})
plt.ylabel("the accuracy")
plt.xlabel("iter number / 100")
plt.show()