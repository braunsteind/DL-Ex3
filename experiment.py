import dynet as dy
import numpy as np

# usage:
EMBED_SIZE = 100
EPOCH = 10
# TODO change
VOCAB = "0123456789abcd"
VOCAB_SIZE = len(VOCAB)
V2I = {char: i for i, char in enumerate(VOCAB)}


# acceptor LSTM
class LstmAcceptor(object):
    def __init__(self, in_dim, lstm_dim, out_dim, model):
        self.builder = dy.VanillaLSTMBuilder(1, in_dim, lstm_dim, model)
        self.W = model.add_parameters((out_dim, lstm_dim))

    def __call__(self, sequence):
        lstm = self.builder.initial_state()
        W = self.W.expr()  # convert the parameter into an Expession (add it to graph)
        outputs = lstm.transduce(sequence)
        result = W * outputs[-1]
        return result


def predict(w):
    dy.renew_cg()  # new computation graph
    vecs = [embeds[V2I[char]] for char in w]
    preds = dy.softmax(acceptor(vecs))
    vals = preds.npvalue()
    return np.argmax(vals)


def accuracy(data):
    good = 0
    bad = 0
    for sequence, label in data:
        y_hat = predict(sequence)
        if y_hat == label:
            good += 1
        else:
            bad += 1
    return float(good) / float(len(data))


# TODO change
def read_file_and_get_data(file_name):
    with open(file_name) as file:
        lines = file.readlines()
        tagged_examples = []
        for line in lines:
            x, y = line.strip('\n').split(" ")
            tagged_examples.append((x, int(y)))

    return tagged_examples


# TODO change
train_file_name = "train"
test_file_name = "test"
train_data = read_file_and_get_data(train_file_name)
test_data = read_file_and_get_data(test_file_name)

m = dy.Model()
trainer = dy.AdamTrainer(m)
embeds = m.add_lookup_parameters((VOCAB_SIZE, EMBED_SIZE))
acceptor = LstmAcceptor(EMBED_SIZE, 100, 2, m)

# training code
print "train"
sum_of_losses = 0.0
for epoch in range(EPOCH):
    for sequence, label in train_data:
        dy.renew_cg()  # new computation graph
        vecs = [embeds[V2I[i]] for i in sequence]
        preds = acceptor(vecs)
        loss = dy.pickneglogsoftmax(preds, label)
        sum_of_losses += loss.npvalue()
        loss.backward()
        trainer.update()
    print "loss: " + str(float(sum_of_losses) / len(train_data)) + " accuracy: " + str(accuracy(train_data))
    print sum_of_losses / len(train_data)
    sum_of_losses = 0.0

# TODO print accuracy
print "\n\nPrediction time!\n"
# prediction code:
for sequence in test_data:
    dy.renew_cg()  # new computation graph
    vecs = [embeds[V2I[i]] for i in sequence]
    preds = dy.softmax(acceptor(vecs))
    vals = preds.npvalue()
    print np.argmax(vals), vals
