# -*- coding: utf-8 -*-

STUDENT = {'name': 'Coral Malachi_Daniel Braunstein',
           'ID': '314882853_312510167'}



CHAR_EMBED_DIM = 20
PREFIX = '*prefix*'
WORD_EMBED_DIM = 64
MLP_SIZE = 16
UNK_SUF = "unk-suffix"
LSTM_DIM = 32


SUFFIX = '*suffix*'
CHAR_LSTM_DIM = 64


UNK_PREF = "unk-prefix"


import dynet as dy
import sys
import numpy as np
import random
from collections import Counter, defaultdict
import sys
import argparse

def create_computation_graph(m_input):
    """

       :param m_input:
       :return:the function build the computation graph according to the chosen model
       """
    if chosen_model == 'a' or chosen_model == 'c':
        res = build_computation_graph_for_a_or_c(m_input)
        return res
    if chosen_model == 'b' or chosen_model == 'd':
        res = build_computation_graph_for_b_or_d(m_input)
        return res


def get_word_rep2(word, cf_init):
    """
    get_word_rep function.
    :param word: requested word.
    :return:
    """
    char_indexes = []
    for char in word:
        if char in vc:
            char_indexes.append(vc[char])
        else:
            char_indexes.append(vc["_UNK_"])
    char_embedding = [CHARS_LOOKUP[indx] for indx in char_indexes]

    # calculate y1,y2,..yn and return yn
    return cf_init.transduce(char_embedding)[-1]

# reads train file. adds start*2, end*2 for each sentence for appropriate windows
# split for words and tags
def read_data(file_name, is_set_ner):
    counter = 0
    sent = []
    sent.append(tuple(('start','start')))
    for line in file(file_name):
        counter += 1
        if (counter%5000 == 0):
            print counter
        if len(line.strip()) == 0:
            #sent.append(tuple(('end','end')))
            yield sent
            sent = []
            sent.append(tuple(('start','start')))
        elif len(line.strip()) == 1:
            continue
        else:
            if(is_set_ner):
                # in ner dataset tab is a saperator
                word_and_tag = line.strip().split("\t")
            else:
                # in pos dataset " " is a saperator
                word_and_tag = line.strip().split(" ")
            word = word_and_tag[0]
            tag = word_and_tag[1]
            sent.append(tuple((word,tag)))

# indexes to data
def convert_indexes_to_words(data):
    # strings to IDs
    L2I = {l:i for i,l in enumerate(data)}
    I2L = {i:l for l,i in L2I.iteritems()}
    return L2I,I2L

def get_word_rep_c(w):
    """

    :param w:a word
    :return:Each word will be represented in  the embeddings+subword representation used in assignment 2.
    """
    unk_prefix =UNK_PREF
    unk_suffix = UNK_SUF
    if len(w) >= 3:
        pref = PREFIX + w[:3]
        suff = SUFFIX + w[-3:]
    else:
        pref = unk_prefix
        suff = unk_suffix
    widx = vw[w] if wc[w] > 5 else UNK
    preidx = vw[pref] if wc[pref] > 5 else vw[unk_prefix]
    suffidx = vw[suff] if wc[suff] > 5 else vw[unk_suffix]
    return [WORDS_LOOKUP[widx], WORDS_LOOKUP[preidx], WORDS_LOOKUP[suffidx]]



def build_computation_graph_for_a_or_c(words):
    # Create a new computation graph - clears the current one and starts a new one
    dy.renew_cg()
    # parameters -> expressions
    # Parameters are things need to be trained.
    # Initialize a parameter vector, and add the parameters to be part of the computation graph.

    # initialize the RNNs
    f_init = fwdRNN.initial_state()  # forward
    b_init = bwdRNN.initial_state()  # backword

    second_forward_initialize = secondfwdRNN.initial_state()
    second_backward_initialize = secondbwdRNN.initial_state()

    # get the word vectors. word_rep(...) returns a 128-dim vector expression for each word.
    wembs = []
    # if the model is a - call the right function to get the match represtention
    if chosen_model == 'a':
        for i, w in enumerate(words):
            # convert word to an embbeding vector
            wembs.append(get_word_rep_a(w))
    if chosen_model == 'c':
        for i, w in enumerate(words):
            word, pre, suff = get_word_rep_c(w)
            wembs.append(word + pre + suff)
    #
    """
    feed word vectors into biLSTM
    transduce takes in a sequence of Expressions, and returns a sequence of Expressions
    """

    # print wembs.__sizeof__()
    fw_exps = f_init.transduce(wembs)  # forward
    bw_exps = b_init.transduce(reversed(wembs))  # backword

    """
         biLSTM states

         Concatenate list of expressions to a single batched expression.
         All input expressions must have the same shape.
    """

    # bi_exps = [dy.concatenate([f, b]) for f, b in zip(fw_exps, reversed(bw_exps))]
    bi_exps = [dy.concatenate([f, b]) for f, b in zip(fw_exps, bw_exps)]

    # print bi_exps.__sizeof__()
    # second BILSTM layer, input: b1,b2..bn, output: b'1,b'2, b'3..
    forward_y_tag = second_forward_initialize.transduce(bi_exps)
    backward_y_tag = second_backward_initialize.transduce(reversed(bi_exps))

    # concat the results
    b_final = [dy.concatenate([y1_tag, y2_tag]) for y1_tag, y2_tag in zip(forward_y_tag, backward_y_tag)]

    # feed each biLSTM state to an MLP
    H = dy.parameter(pH)
    O = dy.parameter(pO)

    exps = []
    for x in b_final:
        r_t = O * (dy.tanh(H * x))
        exps.append(r_t)

    return exps  # results of model


def build_computation_graph_for_b_or_d(words):
    """

        :param words:
        :return: build new computation graph for the models b/d
        """
    # Create a new computation graph - clears the current one and starts a new one
    dy.renew_cg()
    # Parameters are things need to be trained.
    # Initialize a parameter vector, and add the parameters to be part of the computation graph.
    H = dy.parameter(pH)
    O = dy.parameter(pO)

    # initialize the RNNs
    f_init = fwdRNN.initial_state()
    b_init = bwdRNN.initial_state()

    second_forward_initialize = secondfwdRNN.initial_state()
    second_backward_initialize = secondbwdRNN.initial_state()

    cf_init = cFwdRNN.initial_state()
    cb_init = cBwdRNN.initial_state()

    if chosen_model == 'b':
        # get the word vectors. word_rep(...) returns a 128-dim vector expression for each word.
        wembs = [get_word_rep2(w, cf_init) for w in words]
    if chosen_model == 'd':
        wembs = [get_word_rep_d(w, cf_init) for w in words]

    # feed word vectors into biLSTM
    fw_exps = f_init.transduce(wembs)
    bw_exps = b_init.transduce(reversed(wembs))

    # biLSTM states
    bi_exps = [dy.concatenate([f, b]) for f, b in zip(fw_exps, reversed(bw_exps))]

    # second BILSTM layer, input: b1,b2..bn, output: b'1,b'2, b'3..
    forward_y_tag = second_forward_initialize.transduce(bi_exps)
    backward_y_tag = second_backward_initialize.transduce(reversed(bi_exps))

    # concat the results
    b_final = [dy.concatenate([y1_tag, y2_tag]) for y1_tag, y2_tag in zip(forward_y_tag, backward_y_tag)]

    # feed each biLSTM state to an MLP
    exps = []
    for x in b_final:
        r_t = O * (dy.tanh(H * x))
        exps.append(r_t)

    return exps



def get_word_rep_d(w, cf_init):
    # making params for linear layer
    W = dy.parameter(W_d)
    b = dy.parameter(b_d)
    first = get_word_rep_a(w)
    second = get_word_rep2(w, cf_init)
    word_embeddings_d_model = dy.concatenate([first, second])

    # linear layer calculations
    res = ((W * word_embeddings_d_model) + b)
    return res


def calc_loss(words, tags, vecs):
    errs = []
    for v,t in zip(vecs,tags):
        tid = vt[t]
        err = dy.pickneglogsoftmax(v, tid)
        errs.append(err)
    return dy.esum(errs)

def sent_loss(words, tags):
    return calc_loss(words, tags, create_computation_graph(words))

def tag_sent_precalc(words, vecs):
    log_probs = [v.npvalue() for v in vecs]
    tags = []
    for prb in log_probs:
        tag = np.argmax(prb)
        tags.append(ix_to_tag[tag])
    return zip(words, tags)

def tag_sent(words):
    return tag_sent_precalc(words, create_computation_graph(words))

def get_word_rep_a(w):
    """

        :param w:a word
        :return:Each word will be represented as an embedding vector
        """
    widx = vw[w] if wc[w] > 5 else UNK
    return WORDS_LOOKUP[widx]


def get_test_set(test_set):
    """

    :param test_set:
    :return: the function get the test file and turn in into set of lines
    """
    counter = 0
    m_line = []
    m_line.append('start')
    for line in file(test_set):
        counter += 1
        # if (counter%5000 == 0):
        #     print counter
        if len(line.strip()) == 0:
            m_line.append('end')
            yield m_line
            m_line = []
            m_line.append('start')
        else:
            m_word = line.strip()
            m_line.append(m_word)

if __name__ == '__main__':

    chosen_model =  sys.argv[1]
    is_set_ner = False
    if is_set_ner == False:
        train = list(read_data('pos/train', is_set_ner))
    else:
        train = list(read_data('ner/train', is_set_ner))

    if chosen_model == 'a':
        words=[]
        wc = Counter()
        tags=[]

        for sent in train:
            for w,p in sent:
                words.append(w)
                tags.append(p)
                wc[w]+=1
        words.append("_UNK_")

        """
                if the chosen model is b : 
                Each word will be represented in a character-level LSTM

                if the chosen model is d : 
                Each word will be represented in a concatenation of (a) and (b) followed by a linear layer     
                """

    if chosen_model == 'b' or chosen_model == 'd':

        chars=set()
        words=[]
        tags=[]
        wc=Counter()
        for sent in train:
            for w,p in sent:

                words.append(w)
                tags.append(p)
                chars.update(w)

                wc[w]+=1
        words.append("_UNK_")
        chars.add("<*>")
        chars.add("_UNK_")

        vc , ix_to_char = convert_indexes_to_words(set(chars))
        CUNK = vc["_UNK_"]
        nchars  = len(vc)

    """
        if the chosen model is c : 
        Each word will be represented in the embeddings+subword representation used in assignment 2.
        """
    if chosen_model == 'c':
        words=[]
        tags=[]
        wc=Counter()
        # loop each line in train set
        for sent in train:
            # split to word and its tag
            for w,p in sent:
                # add the current word to the words list
                words.append(w)
                if len(w) >= 3:
                    pref = PREFIX + w[:3]
                    suff = SUFFIX + w[-3:]
                tags.append(p)
                wc[w]+=1
        words.append("_UNK_")
        # create prefix and suffix for unknown words and append them to the list
        words.append(UNK_SUF)
        words.append(UNK_PREF)

    # for words - convert words to index and index to words
    vw, ix_to_word = convert_indexes_to_words(set(words))
    # for tags - convert tags to index and index to tags
    vt, ix_to_tag = convert_indexes_to_words(set(tags))

    UNK = vw["_UNK_"]

    len_words = len(vw)
    print len_words

    len_of_tags  = len(vt)
    print (vt)
    print len_of_tags

    #model = dy.Model()
    model = dy.ParameterCollection()
    trainer = dy.AdamTrainer(model)

    if chosen_model == 'c':
        # word embedding matrix
        WORDS_LOOKUP = model.add_lookup_parameters((len_words, WORD_EMBED_DIM))
    else:
        WORDS_LOOKUP = model.add_lookup_parameters((len_words, WORD_EMBED_DIM))

    if chosen_model == 'b' or chosen_model == 'd':
        """
                    if the chosen model is b : 
                    Each word will be represented in a character-level LSTM

                    if the chosen model is d : 
                    Each word will be represented in a concatenation of (a) and (b) followed by a linear layer     
                    """
        CHARS_LOOKUP = model.add_lookup_parameters((nchars, CHAR_EMBED_DIM))

    # MLP on top of biLSTM outputs 100 -> 32 -> len_of_tags
    #pH = model.add_parameters((200, 10*2))
    pH = model.add_parameters((MLP_SIZE, WORD_EMBED_DIM))
    #pO = model.add_parameters((len(set(tags)),  200))
    pO = model.add_parameters((len(set(tags)), MLP_SIZE))

    #model.populate(sys.argv[2])

    if chosen_model == 'c':
        """
                    if the chosen model is c : 
                    Each word will be represented in the embeddings+subword representation used in assignment 2.
                    """
        fwdRNN = dy.LSTMBuilder(1, WORD_EMBED_DIM, LSTM_DIM, model)
        bwdRNN = dy.LSTMBuilder(1, WORD_EMBED_DIM, LSTM_DIM, model)

        secondfwdRNN = dy.LSTMBuilder(1, WORD_EMBED_DIM, LSTM_DIM, model)
        secondbwdRNN = dy.LSTMBuilder(1, WORD_EMBED_DIM, LSTM_DIM, model)

    elif chosen_model == 'd':
        """
               if the chosen model is d :
               a concatenation of (a) and (b) followed by a linear layer.
               that is the reason why the size of the input this time is 100 = 50*2
               """
        fwdRNN = dy.LSTMBuilder(1, WORD_EMBED_DIM, LSTM_DIM, model)
        bwdRNN = dy.LSTMBuilder(1, WORD_EMBED_DIM, LSTM_DIM, model)

        secondfwdRNN = dy.LSTMBuilder(1, WORD_EMBED_DIM, LSTM_DIM, model)
        secondbwdRNN = dy.LSTMBuilder(1, WORD_EMBED_DIM, LSTM_DIM, model)

        cFwdRNN = dy.LSTMBuilder(1, CHAR_EMBED_DIM, CHAR_LSTM_DIM, model)

        W_d = model.add_parameters((WORD_EMBED_DIM, WORD_EMBED_DIM * 2))
        b_d = model.add_parameters((WORD_EMBED_DIM))
        cBwdRNN = dy.LSTMBuilder(1, 50, 25, model)
    else:
        """
                           if the chosen model is b : 
                           Each word will be represented in a character-level LSTM

                           if the chosen model is a : 
                           Each word will be represented in an embedding vector
                           """

        fwdRNN = dy.LSTMBuilder(1, WORD_EMBED_DIM, LSTM_DIM, model)
        bwdRNN = dy.LSTMBuilder(1, WORD_EMBED_DIM, LSTM_DIM, model)

        secondfwdRNN = dy.LSTMBuilder(1, WORD_EMBED_DIM, LSTM_DIM, model)
        secondbwdRNN = dy.LSTMBuilder(1, WORD_EMBED_DIM, LSTM_DIM, model)

    if chosen_model == 'b':
        """
                if the chosen model is b : 
                Each word will be represented in a character-level LSTM
                """

        cFwdRNN = dy.LSTMBuilder(1, CHAR_EMBED_DIM, CHAR_LSTM_DIM, model)
        cBwdRNN = dy.LSTMBuilder(1, CHAR_EMBED_DIM, CHAR_LSTM_DIM, model)
    """
    The first option is to save the complete ParameterCollection object. At loading time, the user 
    should define and allocate the same parameter objects that were present in the model when it was
     saved, and in the same order (this usually amounts to having the same parameter creation called 
     by both code paths), and then call populate on the ParameterCollection object containing the parameters
      that should be loaded.
    """

    model.populate(sys.argv[2])
    """
    create predictions file - predict for each word in the test set its tag
    """

    test_file = open("test4d_after_fix.pos", "w")
    test = get_test_set(sys.argv[3])
    for s in test:
        words = [w for w in s]
        tags = [t for w,t in tag_sent(words)]
        for k in range(1, len(words)-1):
            test_file.write(words[k] + ' ' + tags[k])
            print (words[k] + ' ' + tags[k])
            test_file.write("\n")





