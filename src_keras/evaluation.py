"""Contributors: Kayode Olaleye
"""
# """ Evaluate a model's performance """

import pickle
import os
import numpy as np
from sklearn import metrics
from utils import create_submission_file
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def evaluate_model(model, model_id, comments, list_of_classes, output_dir):
    """ Calculate several metrics for the model and create a visualisation of the test dataset. """
    
    print('_' * 100)
    print('Start evaluating model.')
    x_test= comments
    #X = normalise_input(X)
    #print('Model Metric Names: ', model.metrics_names)
    y_predicted = model.predict(x_test, verbose = 1)
    print(y_predicted[:-1].shape)
    print(y_predicted[0:9])
    path = create_submission_file(list_of_classes, y_predicted, model_id, output_dir)
    print("Model Evaluation completed!\n")
    print("Submission file created and saved to {}".format(path))
    