**Created on**: 18-JAN-2018:03:57am

**Created by**: Kayode Olaleye

**Purpose**: This is where any member of DeepPush or its subgroup can submit a request for a service any other member or subgroup.

**SOP**: To submit a request to this file, use the following format:

BEGIN REQUEST
---------------------

**Request Submitted by**: Kayode Olaleye

**Date**: 18-JAN-2018:04:04am

**Group**: Research/Model

**Members**: Kayode Olaleye, Kabuga Emmanuel

**Response due**: 22-JAN-2018:04:00pm

**DETAILS**:
--------------
To train a model, the following is required.

- Dataset: training set, test set and validation set.
- Validation set is not provided by the owner of the competition. Hence, a 0.1 split of the training set can be made.
- A function named `data_generator` (or any other descriptive name) that preprocesses the dataset and outputs a list of **comments**: x and their corresponding **classes**: y. Where x is a list and y is a numpy array of **one-hot** vectors. 
Example:

```python
def data_generator(dataset, ...):
    ...
    ...
    return [x, y]
```
- As mentioned by @Kabuga on slack, some comments are not labelled. `data_generator` should handle this (we are open to further discussion about this) but you may not handle the padding of the comments, we can do that from the model side. Furthermore, among other things, the function should return dataset that is as clean/filtered as possible.

**Important**: Training a model is an iterative process and usually require  arbitrary back-and-forth movement between data preprocessing and experimentation so bear in mind that we can make more requests after the first is served and tested.

_Cheers_

END REQUEST
------------------------------