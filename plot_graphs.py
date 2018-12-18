import cPickle as pickle
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import collections as co

def load_accuracy_list(pkl_name):
    with open(pkl_name) as dicts_file:
        dict = pickle.load(dicts_file)
    return co.OrderedDict(sorted(dict.items()))

a_ner = load_accuracy_list("a_model_ner.pkl")
b_ner = load_accuracy_list("b_model_ner.pkl")
c_ner = load_accuracy_list("c_model_ner.pkl")
d_ner = load_accuracy_list("d_model_ner.pkl")

a_pos = load_accuracy_list("a_model_pos.pkl")
b_pos = load_accuracy_list("b_model_pos.pkl")
c_pos = load_accuracy_list("c_model_pos.pkl")
d_pos = load_accuracy_list("d_model_pos.pkl")

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