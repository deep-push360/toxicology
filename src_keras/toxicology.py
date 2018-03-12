import argparse
import time 
import os
import sys
import fiona
from config import DATASETS, OUTPUT_DIR, EMBEDDING_FILE
from config import SAMPLE_SUBMISSION_FILE, OUTPUT_DIR
from model_cnn import init_model, train_model, compile_model
from evaluation import evaluate_model
from utils import save_makedirs, save_model_summary, load_model
from utils import create_directories, load_train_data, load_test_data, generate_emb_matrix
import numpy as np

np.random.seed(123)

def create_parser():
    parser = argparse.ArgumentParser(description="Train a convolutional neural network to predict comments class probability")
    
    parser.add_argument(
        "-l", "--load-train",
        dest="load_train",
        action="store_const",
        const=True,
        default=False,
        help="When selected load train data.")
    
    parser.add_argument(
        "-m", "--load-test",
        dest="load_test",
        action="store_const",
        const=True,
        default=False,
        help="When selected load test data.")
    
    parser.add_argument(
        "-i", "--init-model",
        dest="init_model",
        action="store_const",
        const=True,
        default=False,
        help="When selected initialise model.")
    
    parser.add_argument(
        "-t", "--train-model",
        dest="train_model",
        action="store_const",
        const=True,
        default=False,
        help="When selected train model.")
    
    parser.add_argument(
        "-e", "--evaluate-model",
        dest="evaluate_model",
        action="store_const",
        const=True,
        default=False,
        help="When selected evaluate model.")
    
    parser.add_argument(
        "-d", "--debug",
        dest="debug",
        action="store_const",
        const=True,
        default=False,
        help="Run on a small test dataset.")
    
    parser.add_argument(
        "-a", "--architecture",
        dest="architecture",
        default=False,
        choices=["base_arch", "base_glove_arch",  "deep_arch"],
        help="Neural net architecture.")
    
    parser.add_argument(
        "-v", "--visualise",
        dest="visualise",
        action="store_const",
        const=True,
        default=False,
        help="Visualise labels.")
    
    parser.add_argument(
        "-T", "--tensorboard",
        dest="tensorboard",
        action="store_const",
        const=True,
        default=False,
        help="Store tensorboard data while training.")
    
    parser.add_argument(
        "-C", "--checkpoints",
        dest="checkpoints",
        action="store_const",
        const=True,
        default=False,
        help="Create checkpoints while training.")
    
    parser.add_argument(
        "-E", "--earlystop",
        dest="earlystop",
        action="store_const",
        const=True,
        default=False,
        help="Create earlystop while training.")
    
    parser.add_argument(
        "--dataset-train",
        default="train",
        choices=["train", "test"],
        help="Determine which dataset to use for training.")
    
    parser.add_argument(
        "--dataset-test",
        default="test",
        choices=["test"],
        help="Determine which dataset to use for testing.")
    
    parser.add_argument(
        "--classes",
        default = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
        type=list,
        help="List of all classes in the dataset.")
        
    parser.add_argument(
        "--epochs",
        default=1,
        type=int,
        help="Number of training epochs.")
    
    parser.add_argument(
        "--embed-dims",
        default=50,
        type=int,
        help="Size of each word vector.")
    
    parser.add_argument(
        "--model-id",
        default=None,
        type=str,
        help="Model that should be used. must be an existing ID.")
        
    parser.add_argument(
        "--setup",
        default=False,
        action="store_const",
        const=True,
        help="Create all necessary directories for the classifier to work.")
    
    return parser



def main():
    parser = create_parser()
    args = parser.parse_args()
    
    if args.setup:
        create_directories()
        
    if args.train_model or args.evaluate_model or args.load_train:
        train_dataset = DATASETS[args.dataset_train]
        test_dataset = DATASETS[args.dataset_test]
        print('Train dataset path', train_dataset)
        print('Test dataset path', test_dataset)
        comments_train, comments_val, labels_train, labels_val = load_train_data(train_dataset, args.classes)
        print("Training set")
        print("*"*40)
        print('Shape of comment: {}\n'.format(comments_train.shape))
        print('Shape of labels: {}\n'.format(labels_train.shape))
        print("Test set")
        print("*"*40)
        comments_test = load_test_data(test_dataset)
        print("Shape of comment: {}\n".format(comments_test.shape))
        print("*"*40)
        
    if not args.model_id:
        timestamp = time.strftime("%d_%m_%Y_%H%M")
        model_id = "{}_{}".format(timestamp, args.architecture)
    else:
        model_id = args.model_id
        
    if args.init_model or args.train_model or args.evaluate_model:
        #Generating embedding init weights using GloVe.5d
        print("*"*40)
        #train_dataset = DATASETS[args.dataset_train]
        embedding_matrix = generate_emb_matrix(EMBEDDING_FILE, train_dataset, 
                                               args.embed_dims, args.classes)
        model_dir = os.path.join(OUTPUT_DIR, model_id)
        save_makedirs(model_dir)
        
    
    # Hyperparameters for the model. Since there are so many of them it is 
    # more convenient to set them in the source code as opposed to passing 
    # them as arguments to the Command Line Interface. We use a list of tuples instead of a 
    # dict since we want to print the hyperparameters and for that purpose 
    # keep them in the predefined order.

    hyperparameters = [
        ("architecture", args.architecture),
        # Hyperparameter for embedding layer
        ("embedding_dims", 50),
        #Hyperparameter for max_features a.k.a vocab size
        ("vocab_size", 20000),
        #Hyperparameter for sequence length a.k.a max_text_length
        ("sequence_length", 400),
        #Hyperparameters for number of filters
        ("filters", 250),
        #Hyperparameters for filter size
        ("filter_size", 3),
        #Hyperparameter for dense units
        ("hidden_dims", 250),
        # Hyperparameters for the first convolutional layer.
        ("num_filter_1", 512),
        ("filter_size_1", 3),
        # Hyperparameter for the first maxpooling
        ("pool_stride_1", 1),
        # Hyperparameter for the second convolutional layer).
        ("num_filter_2", 512),
        ("filter_size_2", 4),
        # Hyperparameter for the second maxpooling
        ("pool_stride_2", 1),
         # Hyperparameter for the third convolutional layer).
        ("num_filter_3", 512),
        ("filter_size_3", 5),    
        # Hyperparameter for dropout
        ("keepprob", 0.5), 
        # Hyperparameter for the third maxpooling
        ("pool_stride_3", 1),
        # Hyperparameters for Stochastic Gradient Descent.
        ("learning_rate", 0.05),
        ("momentum", 0.9),
        ("decay", 0.0)
        ]
    
    if args.init_model:
        model = init_model(model_id, comments_train, embedding_matrix, **dict(hyperparameters))
        save_model_summary(hyperparameters, model, model_dir)
    elif args.train_model or args.evaluate_model:
        hyperparameters = dict(hyperparameters)
        model = load_model(model_id)
        model = compile_model(model, hyperparameters["learning_rate"], hyperparameters['momentum'], hyperparameters["decay"])
        
    if args.train_model:
        model = train_model(
            model, comments_train, labels_train, comments_val, labels_val,
            model_id, model_dir, nb_epoch = args.epochs,
            checkpoints = args.checkpoints,
            tensorboard = args.tensorboard, earlystop = args.earlystop)
  
    if args.evaluate_model:
        evaluate_model(model, model_id, comments_test, args.classes, OUTPUT_DIR)

if __name__ == '__main__':
    main()
