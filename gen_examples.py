STUDENT = {'name': 'Coral Malachi_Daniel Braunstein',
'ID': '314882853_312510167'}

import random

def make_good_examples(n,add_class):
    pos = []
    for i in range(n):
        randoms = [None]*9
        for k in range(9):
            randoms[k] = random.randint(1,15)

        pos_example = ''
        for j in range(randoms[0]):
            pos_example += str(random.randint(1,9))

        pos_example += 'a'*randoms[1]

        for j in range(randoms[2]):
            pos_example += str(random.randint(1,9))

        pos_example += 'b' * randoms[3]

        for j in range(randoms[4]):
            pos_example += str(random.randint(1,9))

        pos_example += 'c' * randoms[5]

        for j in range(randoms[6]):
            pos_example += str(random.randint(1,9))

        pos_example += 'd' * randoms[7]

        for j in range(randoms[8]):
            pos_example += str(random.randint(1,9))

        if add_class:
            pos.append('1 '+pos_example)
        else:
            pos.append(pos_example)
    return pos


def make_bad_examples(n, add_class):
    pos = []
    for i in range(n):
        randoms = [None] * 9
        for k in range(9):
            randoms[k] = random.randint(1, 15)

        pos_example = ''
        for j in range(randoms[0]):
            pos_example += str(random.randint(1, 9))

        pos_example += 'a' * randoms[1]

        for j in range(randoms[2]):
            pos_example += str(random.randint(1, 9))

        pos_example += 'c' * randoms[3]

        for j in range(randoms[4]):
            pos_example += str(random.randint(1, 9))

        pos_example += 'b' * randoms[5]

        for j in range(randoms[6]):
            pos_example += str(random.randint(1, 9))

        pos_example += 'd' * randoms[7]

        for j in range(randoms[8]):
            pos_example += str(random.randint(1, 9))

        if add_class:
            pos.append('0 ' + pos_example )
        else:
            pos.append(pos_example)
    return pos

def generate_examples():
    pos_examples = make_good_examples(500, False)
    neg_examples = make_bad_examples(500, False)

    with open('pos_examples', 'w') as file1:
        for pos_example in pos_examples:
            file1.write("{0}\n".format(pos_example))

    with open('neg_examples', 'w') as file2:
        for neg_example in neg_examples:
            file2.write("{0}\n".format(neg_example))

def make_train_set():

    pos_examples = make_good_examples(2000, True)
    neg_examples = make_bad_examples(2000, True)

    train_set = pos_examples + neg_examples
    random.shuffle(train_set)

    return train_set

def make_test_set():

    pos_examples = make_good_examples(200, True)
    neg_examples = make_bad_examples(200, True)

    test_set = pos_examples + neg_examples
    random.shuffle(test_set)
    return test_set


def make_test_and_train_sets():
    train = make_train_set()

    test = make_test_set()

    with open('train', 'w') as file1:
        file1.write("\n".join(train))

    with open('test', 'w') as file2:
        file2.write("\n".join(test))

    # with open('test', 'w') as file2:
    #     for x in test:
    #         file2.write("{0}\n".format(x))

if __name__ == '__main__':
    generate_examples()
    make_test_and_train_sets()




