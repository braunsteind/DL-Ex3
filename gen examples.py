import rstr
from random import shuffle

# generate pos examples
with open("pos_examples", "w") as file:
    examples = []
    for i in range(500):
        examples.append(
            rstr.xeger(r'[1-9]{1,10}a{1,10}[1-9]{1,10}b{1,10}[1-9]{1,10}c{1,10}[1-9]{1,10}d{1,10}[1-9]{1,10}'))
    file.write("\n".join(examples))

# generate neg examples
with open("neg_examples", "w") as file:
    examples = []
    for i in range(500):
        examples.append(
            rstr.xeger(r'[1-9]{1,10}a{1,10}[1-9]{1,10}c{1,10}[1-9]{1,10}b{1,10}[1-9]{1,10}d{1,10}[1-9]{1,10}'))
    file.write("\n".join(examples))

# TODO need the change
examples = []
with open("neg_examples", "r") as neg_file, open("pos_examples", "r") as pos_file:
    pos_content, neg_content = pos_file.readlines(), neg_file.readlines()
    pos_content = [example.strip('\n') + " 1" for example in pos_content]
    neg_content = [example.strip('\n') + " 0" for example in neg_content]
    examples += pos_content + neg_content
shuffle(examples)
test_list, train_list = examples[:200], examples[200:]
with open("train", "w") as train_file, open("test", "w") as test_file:
    train_file.write("\n".join(train_list))
    test_file.write("\n".join(test_list))
