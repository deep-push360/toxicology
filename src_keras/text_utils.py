"""Contributors: Kayode Olaleye"""

import numpy as np
import re
import itertools
from collections import Counter
from preprocessing import data_generator

def pad_comments(comments, padding_word="P"):
    """
    Pads all comments to the same length. The length is defined by the longest comment.
    Returns padded comments.
    """
    sequence_length = max(len(x) for x in comments)
    padded_comments = []
    for i in range(len(comments)):
        comment = comments[i]
        num_padding = sequence_length - len(comment)
        new_comment = comment + padding_word * num_padding
        padded_comments.append(new_comment)
    return padded_comments

def build_vocab(comments):
    """
    Builds a vocabulary mapping from word to index based on the comments.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*comments))
    # Mapping from index to word
    vocab_inv = [x[0] for x in word_counts.most_common()]
    vocab_inv = list(sorted(vocab_inv))
    # Mapping from word to index
    vocab = {x: i for i, x in enumerate(vocab_inv)}
    return [vocab, vocab_inv]

def generate_input_data(comments, classes, vocab):
    """
    Maps comments and classes to vectors based on a vocabulary.
    """
    x = np.array([[vocab[word] for word in comment] for comment in comments])
    y = np.array(classes)
    return [x, y]

def load_keras_data(path_to_file):
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, classes, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    comments, classes = data_generator(path_to_file)
    comments_padded = pad_comments(comments)
    vocabulary, vocabulary_inv = build_vocab(comments_padded)
    x, y = generate_input_data(comments_padded, classes, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]
