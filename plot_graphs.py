import cPickle as pickle
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import collections as co

def load_accuracy_list(pkl_name):
    with open(pkl_name) as dicts_file:
        dict = pickle.load(dicts_file)
    return co.OrderedDict(sorted(dict.items()))

a_ner = load_accuracy_list("a_model_ner.pkl")

label1, = plt.plot(a_ner.keys(), a_ner.values(), "b-", label='a - ner')
plt.legend(handler_map={label1: HandlerLine2D(numpoints=4)})
plt.ylabel("accuracy")
plt.xlabel("iter number / 100")
plt.show()