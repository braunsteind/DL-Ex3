import dynet as dy
import numpy as np

# usage:
VOCAB_SIZE = 1000
EMBED_SIZE = 100
EPOCH = 10


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


# TODO load the data

m = dy.Model()
trainer = dy.AdamTrainer(m)
embeds = m.add_lookup_parameters((VOCAB_SIZE, EMBED_SIZE))
acceptor = LstmAcceptor(EMBED_SIZE, 100, 3, m)

# training code
sum_of_losses = 0.0
for epoch in range(EPOCH):
    # TODO change to train data
    for sequence, label in [((1, 4, 5, 1), 1), ((42, 1), 2), ((56, 2, 17), 1)]:
        dy.renew_cg()  # new computation graph
        vecs = [embeds[i] for i in sequence]
        preds = acceptor(vecs)
        loss = dy.pickneglogsoftmax(preds, label)
        sum_of_losses += loss.npvalue()
        loss.backward()
        trainer.update()
    # TODO change 3 to len(train_data)
    print sum_of_losses / 3
    sum_of_losses = 0.0

print "\n\nPrediction time!\n"
# prediction code:
# TODO change to test data
for sequence in [(1, 4, 12, 1), (42, 2), (56, 2, 17)]:
    dy.renew_cg()  # new computation graph
    vecs = [embeds[i] for i in sequence]
    preds = dy.softmax(acceptor(vecs))
    vals = preds.npvalue()
    print np.argmax(vals), vals
