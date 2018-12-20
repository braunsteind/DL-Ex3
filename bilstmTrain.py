# -*- coding: utf-8 -*-

STUDENT = {'name': 'Coral Malachi_Daniel Braunstein',
           'ID': '314882853_312510167'}

#lets define some consts :
CHAR_EMBED_DIM = 20
PREFIX = '*prefix*'
LSTM_DIM = 32
SUFFIX = '*suffix*'
WORD_EMBED_DIM = 64
UNK_PREF = "unk-prefix"
MLP_SIZE = 16
CHAR_LSTM_DIM = 64
UNK_SUF = "unk-suffix"




import dynet as dy
import sys
import numpy as np
import random
from collections import Counter, defaultdict
import sys
import argparse
import time
from shutil import copyfile
import cPickle as pickle


def get_tags(words, vecs):
    """

    :param words:
    :param vecs:
    :return: the function predict the right tag of word
    """
    log_probs = [v.npvalue() for v in vecs]
    tags = []
    for prb in log_probs:
        tag = np.argmax(prb)
        tags.append(ix_to_tag[tag])
    return tags


def split_line(line):
    """

    :param line:
    :return: the function get a line as an input and return thw words in this line and their tags
    """

    words_after_split = [word for word, tag in line]
    tags_after_split = [tag for word, tag in line]
    return words_after_split, tags_after_split

def convert_indexes_to_words(data):

    """
    :param data:
    :return: the function return representation of word as index
    and of index as words
    """

    #convert labels to indexes
    labels_to_indexes = {l: i for i, l in enumerate(data)}
    #convert indexes to lables
    Index_to_labels = {i: l for l, i in labels_to_indexes.iteritems()}
    return labels_to_indexes, Index_to_labels

def calculate_accuracy(tagged_data, type):
    """

    :param tagged_data:
    :param type:
    :return: the function count the number of correct predictions. the accuracy is the
            count of correct preds / total number of predictions
    """
    total_words = 0
    correct_prediction_count = 0

    for tagged_sentence in tagged_data:
        words, tags = split_line(tagged_sentence)
        #preds = model.get_prediction_on_sentence(words)
        preds = get_tags(words,create_computation_graph(words))
        for pred, tag in zip(preds, tags):
            if type == "pos":
                if pred == tag:
                    correct_prediction_count += 1
                total_words += 1
            # we don't consider correct taggings of Other ("O") label on
            # ner data as correct_prediction_count predictions
            if type == "ner":
                if pred == "O" and tag == "O":
                    continue
                elif pred == tag:
                    correct_prediction_count += 1
                total_words += 1
        #return the accuracy
    return float(correct_prediction_count) / float(total_words) * 100


# reads train file. adds start*2, end*2 for each sentence for appropriate windows
# split for words and tags
def read_data(file_name, is_ner):
    """

    :param file_name:
    :param is_ner: indicates about the dataset
    :return: the function reads the train file. adds start*2, end*2 for each line for appropriate windows
    """
    #copyfile(file_name, 'copy.txt')

    counter = 0
    sent = []
    sent.append(tuple(('start', 'start')))
    for line in file(file_name):
        counter += 1
        if (counter % 5000 == 0):
            print counter
        #when line is empty we finished read the seq
        if len(line.strip()) == 0:
            #sent.append(tuple(('end', 'end')))
            yield sent
            sent = []
            sent.append(tuple(('start', 'start')))
        elif len(line.strip()) == 1:
            continue
        else:
            if (is_ner):
                #in ner dataset tab is a saperator
                word_and_tag = line.strip().split("\t")
            else:
                # in pos dataset " " is a saperator
                word_and_tag = line.strip().split(" ")
            word = word_and_tag[0]
            tag = word_and_tag[1]
            sent.append(tuple((word, tag)))







def create_computation_graph(m_input):
    """

    :param m_input:
    :return:the function build the computation graph according to the chosen model
    """
    if chosen_model == 'c' or chosen_model == 'a':
        res = build_computation_graph_for_a_or_c(m_input)
        return res
    if chosen_model == 'b' or chosen_model == 'd':
        res = build_computation_graph_for_b_or_d(m_input)
        return res


def build_computation_graph_for_a_or_c(words):
    """

    :param words:
    :return: build new computation graph for the models a/c
    """
    #Create a new computation graph - clears the current one and starts a new one
    dy.renew_cg()

    #Parameters are things need to be trained.
    #Initialize a parameter vector, and add the parameters to be part of the computation graph.


    # initialize the RNNs
    f_init = fwdRNN.initial_state()#forward
    b_init = bwdRNN.initial_state()#backword

    second_forward_initialize = secondfwdRNN.initial_state()
    second_backward_initialize = secondbwdRNN.initial_state()

    # get the word vectors. word_rep(...) returns a 128-dim vector expression for each word.
    wembs = []
    #if the model is a - call the right function to get the match represtention
    if chosen_model == 'a':
        for i, w in enumerate(words):
            #convert word to an embbeding vector
            wembs.append(get_word_rep_a(w))
            # if the model is c - call the right function to get the match represtention
    if chosen_model == 'c':
        for i, w in enumerate(words):
            word, pre, suff = word_rep_3(w)
            wembs.append(word + pre + suff)
    #
    """
    feed word vectors into biLSTM
    transduce takes in a sequence of Expressions, and returns a sequence of Expressions
    """

    #print wembs.__sizeof__()
    fw_exps = f_init.transduce(wembs)#forward
    bw_exps = b_init.transduce(reversed(wembs))#backword

    """
         biLSTM states
         
         Concatenate list of expressions to a single batched expression.
         All input expressions must have the same shape.
    """

    # bi_first = [dy.concatenate([f, b]) for f, b in zip(fw_exps, reversed(bw_exps))]
    bi_first = [dy.concatenate([f, b]) for f, b in zip(fw_exps, bw_exps)]

    #print bi_first.__sizeof__()
    # second BILSTM layer, input: b1,b2..bn, output: b'1,b'2, b'3..
    forward_y_tag = second_forward_initialize.transduce(bi_first)
    backward_y_tag = second_backward_initialize.transduce(reversed(bi_first))

    # concat the results
    b_tag = [dy.concatenate([y1_tag, y2_tag]) for y1_tag, y2_tag in zip(forward_y_tag, backward_y_tag)]

    # feed each biLSTM state to an MLP
    H = dy.parameter(pH)
    O = dy.parameter(pO)

    exps = []
    for x in b_tag:
        r_t = O * (dy.tanh(H * x))
        exps.append(r_t)

    return exps#results of model


def build_computation_graph_for_b_or_d(words):
    """

        :param words:
        :return: build new computation graph for the models b/d
        """
    dy.renew_cg()
    # parameters -> expressions
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
    bi_first = [dy.concatenate([f, b]) for f, b in zip(fw_exps, reversed(bw_exps))]

    # second BILSTM layer, input: b1,b2..bn, output: b'1,b'2, b'3..
    forward_y_tag = second_forward_initialize.transduce(bi_first)
    backward_y_tag = second_backward_initialize.transduce(reversed(bi_first))

    # concat the results
    b_tag = [dy.concatenate([y1_tag, y2_tag]) for y1_tag, y2_tag in zip(forward_y_tag, backward_y_tag)]

    # feed each biLSTM state to an MLP
    exps = []
    for x in b_tag:
        ans = O * (dy.tanh(H * x))
        exps.append(ans)

    return exps

# def get_word_rep_a(w):
#     if w in vw:
#         return WORDS_LOOKUP[vw[w]]
#     else:
#         return UNK

def get_word_rep_a(w):
    """

    :param w: current word
    :return: the word represented by chosen model - a
    """
    word_index = vw[w] if wc[w] > 5 else UNK
    return WORDS_LOOKUP[word_index]

def get_word_rep2(word,cf_init):
        """
        get_word_rep function.
        :param word: current word
        :return:the word represented by chosen model - b
        """
        char_indexes = []
        for char in word:
            if char in vc:
                char_indexes.append(vc[char])
            else:
                char_indexes.append(vc["_UNK_"])
        vec_char_embedding = [CHARS_LOOKUP[i] for i in char_indexes]

        # calculate y1,...yn and return yn
        return cf_init.transduce(vec_char_embedding)[-1]



def word_rep_3(w):
    """

    :param word: current word
    :return:the word represented by chosen model - c
    """
    unk_prefix = UNK_PREF
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


def get_word_rep_d(w, cf_init):
    """

        :param word: current word
        :return:the word represented by chosen model - d
        """
    # get params for linear layer
    W = dy.parameter(W_d)
    b = dy.parameter(b_d)
    #get the representation of each model a and b
    first = get_word_rep_a(w)
    second = get_word_rep2(w, cf_init)
    #a concatenation of (a) and (b)
    word_embeddings_d_model =  dy.concatenate([first, second])

    # followed by a linear layer
    res = ((W * word_embeddings_d_model) + b)
    return res


def calc_loss(words, tags, vecs):
    #define an empty list
    m_losses = []
    for v, t in zip(vecs, tags):
        tid = vt[t]
        #compute loss
        err = dy.pickneglogsoftmax(v, tid)
        #append current loss to list
        m_losses.append(err)
    return dy.esum(m_losses)


def sent_loss(words, tags):
    return calc_loss(words, tags, create_computation_graph(words))



if __name__ == '__main__':
    #set train to be pos/ner
    is_ner = False
    start = time.time()
    #get the kind of model - a/b/c/d
    chosen_model = sys.argv[1]
    #read the train data
    train = list(read_data(sys.argv[2], is_ner))
    #read the dev data
    dev = list(read_data("pos/dev",is_ner))

    """
    if the chosen model is a : 
    Each word will be represented in an embedding vector
    """

    if chosen_model == 'a':
        tags = []
        words = []

        wc = Counter()
        #loop each line in train set
        for sent in train:
            #split to word and its tag
            for word_C, match_tag in sent:
                #add the current word to the words list
                words.append(word_C)
                # add the current tag to the words tags
                tags.append(match_tag)
                #update the counter of current word
                wc[word_C] += 1
        #add the "_UNK_" word to the words list for words we wont see in our training set but will see in the dev/test set
        words.append("_UNK_")

    """
        if the chosen model is b : 
        Each word will be represented in a character-level LSTM
        
        if the chosen model is d : 
        Each word will be represented in a concatenation of (a) and (b) followed by a linear layer     
        """

    if chosen_model == 'b' or chosen_model == 'd':
        chars = set()
        words = []
        tags = []
        wc = Counter()
        # loop each line in train set
        for sent in train:
            # split to word and its tag
            for w, p in sent:
                # add the current word to the words list
                words.append(w)
                # add the current tag to the words tags
                tags.append(p)
                chars.update(w)

                wc[w] += 1
        words.append("_UNK_")
        chars.add("<*>")
        chars.add("_UNK_")

        vc, ix_to_char = convert_indexes_to_words(set(chars))
        # add the "_UNK_" word to the vector chars list for words we wont see in our training set but will see in the dev/test set
        CUNK = vc["_UNK_"]
        #get len of chars vectors list
        nchars = len(vc)



    """
    if the chosen model is c : 
    Each word will be represented in the embeddings+subword representation used in assignment 2.
    """

    if chosen_model == 'c':
        words = []
        tags = []
        wc = Counter()
        # loop each line in train set
        for sent in train:
            # split to word and its tag
            for w, p in sent:
                # add the current word to the words list
                words.append(w)
                if len(w) >= 3:
                    pref = PREFIX + w[:3]
                    suff = SUFFIX + w[-3:]
                tags.append(p)
                wc[w] += 1
        words.append("_UNK_")
        #create prefix and suffix for unknown words and append them to the list
        words.append(UNK_SUF)
        words.append(UNK_PREF)

    #for words - convert words to index and index to words
    vw, ix_to_word = convert_indexes_to_words(set(words))
    # for tags - convert tags to index and index to tags
    vt, ix_to_tag = convert_indexes_to_words(set(tags))

    #get index of word _UNK_ and save it in UNK varible
    UNK = vw["_UNK_"]

    #get number of different words
    nwords = len(vw)
    #print nwords


    # get number of different tags
    ntags = len(vt)
    print (vt)
    print ntags

    #init a model with dynet library
    #model = dy.Model()
    model = dy.ParameterCollection()
    #Create an Adam trainer to update the model's parameters.
    trainer = dy.AdamTrainer(model)

    if chosen_model == 'c':
        # word embedding matrix
        WORDS_LOOKUP = model.add_lookup_parameters((nwords, WORD_EMBED_DIM))
    else:
        # word embedding matrix
        WORDS_LOOKUP = model.add_lookup_parameters((nwords, WORD_EMBED_DIM))


    if chosen_model == 'b' or chosen_model == 'd':
        """
            if the chosen model is b : 
            Each word will be represented in a character-level LSTM

            if the chosen model is d : 
            Each word will be represented in a concatenation of (a) and (b) followed by a linear layer     
            """
        CHARS_LOOKUP = model.add_lookup_parameters((nchars, CHAR_EMBED_DIM))

    # MLP on top of biLSTM


    pH = model.add_parameters((MLP_SIZE, WORD_EMBED_DIM))
    #w2 parameter size of number of tags x hidden layer
    pO = model.add_parameters((len(set(tags)), MLP_SIZE))

    if chosen_model == 'c':
        """
            if the chosen model is c : 
            Each word will be represented in the embeddings+subword representation used in assignment 2.
            """
        # word-level LSTMs
        """
        VanillaLSTM allows the creation of a “standard” LSTM,
         ie with decoupled input and forget gates and no peephole connections. 
        """

        # first BILSTM - input: x1,..xn, output: b1,..bn
        fwdRNN = dy.LSTMBuilder(1, WORD_EMBED_DIM, LSTM_DIM, model)  # layers, in-dim, out-dim, model
        bwdRNN = dy.LSTMBuilder(1, WORD_EMBED_DIM, LSTM_DIM, model)

        secondfwdRNN = dy.LSTMBuilder(1, WORD_EMBED_DIM, LSTM_DIM, model)  # layers, in-dim, out-dim, model
        secondbwdRNN = dy.LSTMBuilder(1, WORD_EMBED_DIM, LSTM_DIM, model)
    elif chosen_model == 'd':
        """
        if the chosen model is d :
        a concatenation of (a) and (b) followed by a linear layer.
        that is the reason why the size of the input this time is 100 = 50*2
        """
        fwdRNN = dy.LSTMBuilder(1, WORD_EMBED_DIM, LSTM_DIM, model)  # layers, in-dim, out-dim, model
        bwdRNN = dy.LSTMBuilder(1, WORD_EMBED_DIM, LSTM_DIM, model)

        secondfwdRNN = dy.LSTMBuilder(1, WORD_EMBED_DIM, LSTM_DIM, model)  # layers, in-dim, out-dim, model
        secondbwdRNN = dy.LSTMBuilder(1, WORD_EMBED_DIM, LSTM_DIM, model)


        cFwdRNN = dy.LSTMBuilder(1, CHAR_EMBED_DIM, CHAR_LSTM_DIM, model)

        W_d = model.add_parameters((WORD_EMBED_DIM, WORD_EMBED_DIM * 2))
        b_d = model.add_parameters((WORD_EMBED_DIM))
        cBwdRNN = dy.LSTMBuilder(1, 50, 25, model)
    else:
        #a/b model
        # word-level LSTMs
        fwdRNN = dy.LSTMBuilder(1, WORD_EMBED_DIM, LSTM_DIM, model)  # layers, in-dim, out-dim, model
        bwdRNN = dy.LSTMBuilder(1, WORD_EMBED_DIM, LSTM_DIM, model)

        secondfwdRNN = dy.LSTMBuilder(1, WORD_EMBED_DIM, LSTM_DIM, model)  # layers, in-dim, out-dim, model
        secondbwdRNN = dy.LSTMBuilder(1, WORD_EMBED_DIM, LSTM_DIM, model)

    if chosen_model == 'b':
        # char-level LSTMs
        cFwdRNN = dy.LSTMBuilder(1, CHAR_EMBED_DIM, CHAR_LSTM_DIM, model)
        cBwdRNN = dy.LSTMBuilder(1, CHAR_EMBED_DIM, CHAR_LSTM_DIM, model)

    print ("start time: %r" % (time.time() - start))
    start = time.time()

    acc = []
    i = all_time = all_tagged = this_tagged = this_loss = 0
    #save the accuracy results for ploting the graph after
    graph = {}
    for epoch_number in range(5):
        # random.shuffle(train)
        for s in train:
            i += 1
            print "aaaaaaaaaaaaaaaaaa"
            if i % 500 == 0:  # print status
                acc = calculate_accuracy(dev,"pos")
                trainer.status()
                #print(this_loss / this_tagged)
                all_tagged += this_tagged
                this_loss = this_tagged = 0
                all_time = time.time() - start
                graph[i / 100] = acc

            """
            split the current line to words and tags
            """
            words = [w for w, t in s]
            m_class = [t for w, t in s]

            #calculate the loss
            print "hereee"
            loss_exp = sent_loss(words, m_class)
            print "hiiii"
            my_loss = loss_exp.scalar_value()
            this_loss += my_loss
            this_tagged += len(m_class)
            #performs back-propagation, and accumulates the gradients of the parameters
            loss_exp.backward()
            #updates parameters of the parameter collection that was passed to its constructor.
            trainer.update()
        print("epoch %r is done" % epoch_number)
        trainer.update_epoch(1.0)

#save results for ploting the needed graphs later
    with open(chosen_model + "_model_" + 'pos_fixed' + ".pkl", "wb") as output:
        pickle.dump(graph, output, pickle.HIGHEST_PROTOCOL)

    #gets a base filename and a list of saveable objects, and saves them to file.
    model.save(sys.argv[3])