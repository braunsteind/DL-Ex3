# -*- coding: utf-8 -*-


STUDENT = {'name': 'Coral Malachi_Daniel Braunstein',
           'ID': '314882853_312510167'}
"""
    in this file we'll implement acceptor LSTM,
    that reads in a sequence of vectors, passes the final vector through a
    linear layer followed by a softmax, and produces an output.
"""

from time import time

import dynet as dy
import numpy as np

global LAYERS
global INPUT_DIM
global HIDDEN_DIM
global vocab
global VOCAB_SIZE
global OUTPUT_DIM
global num_epochs

LAYERS = 2
INPUT_DIM = 30
HIDDEN_DIM = 35

part = 2
if part == 1:
    vocab = ['a', 'b', 'c', 'd', '1', '2', '3', '4', '5', '6', '7', '8', '9']
else:
    vocab = [str(i) for i in xrange(10)] + [chr(i) for i in xrange(ord('a'), ord('z'))] + [chr(i) for i in
                                                                                           xrange(ord('A'), ord('Z'))]
VOCAB_SIZE = len(vocab)
OUTPUT_DIM = 2
num_epochs = 6

model = dy.Model()

# creates an LSTM unit
lstm = dy.LSTMBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, model)

lookup = model.add_lookup_parameters((VOCAB_SIZE, INPUT_DIM))

# MLP after LSTM outputs
W_param = model.add_parameters((OUTPUT_DIM, HIDDEN_DIM))
bias_param = model.add_parameters(OUTPUT_DIM)


def get_data(file_name):
    """

    :param file:
    :return: the function return the data of the file
    """
    data_set = []
    # loop the text and create a set of words
    for line in file(file_name):
        read_line = line.strip()
        data_set.append(read_line)
    return data_set


def create_set_loader(dataset, dict):
    new_set = []
    for line in dataset:
        # define empty array
        new_row = []
        # split line to word and tag
        m_class, m_word = line.split(' ')

        if m_class == '0':
            m_class = 0
        else:
            m_class = 1

        new_row.append(m_class)
        # each character is given an index that represents an embedding vector
        for x in range(len(m_word)):
            new_row.append(dict[m_word[x]])
        new_set.append(new_row)
    return new_set


def convert_vocab_to_index(vocab):
    words_as_index = {x: i for i, x in enumerate(vocab)}
    return words_as_index


def compute_loss(word, tag):
    """

    :param word:
    :param tag:
    :return: return compute loss of RNN for one sentence
    """
    """
        The ComputationGraph is composed of expressions, which relate to the inputs and outputs of the network,
        as well as the Parameters of the network.
        When dynet is imported, a new ComputationGraph is created. We can then reset the computation graph to
        a new state by calling renew_cg().
        we call dy.renew_cg() before each sentence -because we want to have a new graph (new network) for
        this sentence
    """

    dy.renew_cg()
    s0 = lstm.initial_state()

    s = s0

    # in rnn each word feeds the network - one after the other
    for char in word:
        # feed network with nect digit
        s = s.add_input(lookup[char])

    # MLP
    W = dy.parameter(W_param)
    b = dy.parameter(bias_param)

    probs = dy.softmax(W * s.output() + b)

    # loss of the model
    """
    . The model will be optimized to minimize the value of the final function in the computation graph
    """
    loss = -dy.log(dy.pick(probs, tag))

    """
    “backward” performs back-propagation, and accumulates the gradients of the parameters within 
    the ParameterCollection data structure.
    """

    loss.backward()
    # updates parameters of the parameter collection that was passed to its constructor.
    trainer.update()
    return loss


# make prediction
def make_prediction(word, tag):
    """

    :param word:
    :param tag:
    :return: the function return the prediction of the model and its loss
    """

    dy.renew_cg()
    s0 = lstm.initial_state()

    # define model parameters
    W = dy.parameter(W_param)
    bias = dy.parameter(bias_param)
    s = s0
    for char in word:
        s = s.add_input(lookup[char])

    probs = dy.softmax(W * s.output() + bias)

    prediction = np.argmax(probs.npvalue())
    # calculate loss
    loss = -dy.log(dy.pick(probs, tag))
    return prediction, loss


def compute_accueacy(set):
    """

    :param set:
    :return: the function return the accuracy of the model
    """
    correct_count = 0
    wrong_pred = 0
    for line in set:
        y, x = line[0], line[1:]
        prediction, loss = make_prediction(x, y)
        if prediction == y:
            correct_count += 1
        else:
            wrong_pred += 1
    return float(correct_count) / float(len(set))


if __name__ == '__main__':
    total_loss_train = 0.0
    total_loss_test = 0.0

    train = get_data('train')
    test = get_data('test')

    vocab_as_index = convert_vocab_to_index(vocab)

    train_data = create_set_loader(train, vocab_as_index)
    test_data = create_set_loader(test, vocab_as_index)

    # Create an Adam trainer to update its parameters.
    trainer = dy.AdamTrainer(model)
    # counting time as required
    begin_time = time()
    for epoch in range(num_epochs):
        print "epoch number: " + str(epoch)

        for line in train_data:
            tag, word = line[0], line[1:]
            loss = compute_loss(word, tag)
            loss_value = loss.value()
            total_loss_train += loss_value
            loss.backward()
            trainer.update()

        print "train loss: " + str(float(total_loss_train) / float(len(train_data)))
        print "train accuracy : " + str(compute_accueacy(train_data))

        correct_answers = 0.0
        for line in test_data:
            tag, word = line[0], line[1:]
            # setup the sentence
            prediction, m_loss = make_prediction(word, tag)

            # m_loss = dy.pickneglogsoftmax(prediction,tag)
            total_loss_test += m_loss.npvalue()
            if tag == prediction:
                correct_answers += 1
        print "test loss: " + str(float(total_loss_test) / float(len(test_data)))
        print "test accuracy: " + str(float(correct_answers) / float(len(test_data)))
        print "number of correct answer: " + str(correct_answers)
        print(len(test_data))
        total_loss_test = 0.0
        total_loss_train = 0.0
        until_now_time = time()
        print "process time until now is: " + str(until_now_time - begin_time)
    end_time = time()
    process_time = end_time - begin_time
    print "process time is: " + str(process_time)
