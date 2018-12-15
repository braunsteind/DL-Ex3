import random

VOCAB = [str(i) for i in xrange(10)] + [chr(i) for i in xrange(ord('a'), ord('z'))] + [chr(i) for i in
                                                                                       xrange(ord('A'), ord('Z'))]
WORD_MAX_LEN = 30


def generate_word():
    """
    Generate random word
    :return: Random word
    """
    # random length for the word
    length = random.randint(1, WORD_MAX_LEN)
    # set empty word
    word = ''
    # build the word
    for _ in xrange(length):
        word += VOCAB[random.randint(0, len(VOCAB) - 1)]
    return word


def generate_palindrome():
    """
    Generate a good palindrome
    :return: Palindrome
    """
    word = generate_word()
    # build palindrome from the word
    palindrome = word + word[::-1]
    return palindrome


def make_good_examples(n, add_class):
    good_examples = []
    for _ in xrange(n):
        example = generate_palindrome()
        if add_class:
            good_examples.append('1 ' + example)
        else:
            good_examples.append(example)
    return good_examples


def make_bad_examples(n, add_class):
    bad_examples = []
    for _ in xrange(n):
        palindrome = generate_palindrome()
        word = generate_word()
        # random index
        index = random.randint(0, len(palindrome) - 1)
        # add the word to the palindrome
        example = palindrome[:index] + word + palindrome[index:]
        # if the example is palindrome
        while example == example[::-1]:
            # generate new word
            word = generate_word()
            # create new example
            example = palindrome[:index] + word + palindrome[index:]
        # add the word inside the palindrome
        if add_class:
            bad_examples.append('0 ' + example)
        else:
            bad_examples.append(example)
    return bad_examples


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


if __name__ == '__main__':
    generate_examples()
    make_test_and_train_sets()
