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
            pos.append(pos_example+' 1')
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
            pos.append(pos_example + ' 1')
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

if __name__ == '__main__':
    generate_examples()



# import rstr
# from random import shuffle
#
# # generate pos examples
# with open("pos_examples", "w") as file:
#     examples = []
#     for i in range(500):
#         examples.append(
#             rstr.xeger(r'[1-9]{1,10}a{1,10}[1-9]{1,10}b{1,10}[1-9]{1,10}c{1,10}[1-9]{1,10}d{1,10}[1-9]{1,10}'))
#     file.write("\n".join(examples))
#
# # generate neg examples
# with open("neg_examples", "w") as file:
#     examples = []
#     for i in range(500):
#         examples.append(
#             rstr.xeger(r'[1-9]{1,10}a{1,10}[1-9]{1,10}c{1,10}[1-9]{1,10}b{1,10}[1-9]{1,10}d{1,10}[1-9]{1,10}'))
#     file.write("\n".join(examples))
#
# # TODO need the change
# examples = []
# with open("neg_examples", "r") as neg_file, open("pos_examples", "r") as pos_file:
#     pos_content, neg_content = pos_file.readlines(), neg_file.readlines()
#     pos_content = [example.strip('\n') + " 1" for example in pos_content]
#     neg_content = [example.strip('\n') + " 0" for example in neg_content]
#     examples += pos_content + neg_content
# shuffle(examples)
# test_list, train_list = examples[:200], examples[200:]
# with open("train", "w") as train_file, open("test", "w") as test_file:
#     train_file.write("\n".join(train_list))
#     test_file.write("\n".join(test_list))
