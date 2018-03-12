""" Contributors: Kayode Olaleye

This sets up the directories where the program can find the data """


import os
from os.path import join

BASE_DIR = join(os.environ["HOME"], "lustre", "random", "playground", "toxicology", "data")
print('base dir: ', BASE_DIR)

#Directories to store everything related to the training data.
#CACHE_DATA_DIR = join(BASE_DIR, "working", "cache")
#CACHE_DIR = join(TRAIN_DATA_DIR, "cache")
#print('cache dir: ', CACHE_DATA_DIR)

#Directories to store the models and weights
MODELS_DIR = join(BASE_DIR, "working", "models")
print('models dir: ', MODELS_DIR)

#Directories for model output (models, visualisations, etc)
OUTPUT_DIR = join(BASE_DIR, "output")
TENSORBOARD_DIR = join(OUTPUT_DIR, "tensorboard")
SUBMISSION_DIR = join(OUTPUT_DIR, "submission")

EMBEDDING_FILE = join(BASE_DIR, 'input', 'glove.6B.50d.txt')
TRAIN_DATA_FILE = join(BASE_DIR, 'input', 'train.csv')
TEST_DATA_FILE = join(BASE_DIR, 'input', 'test.csv')

DATASETS = {
    "train": TRAIN_DATA_FILE,
    "test": TEST_DATA_FILE
    }

SAMPLE_SUBMISSION_FILE = join(BASE_DIR, "input", "sample_submission.csv")
