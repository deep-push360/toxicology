"""Contributors: Zoe Hamel, Samuel Mensah

***IMPORTANT**
Use 
"""

def clean_punctuation(dataset):
   """
   a function that removes all punctuation
   this function takes in dataset as argument.
   where dataset is a csv file

   library needed for this function is pandas.
   """
   dataset = pd.read_csv(dataset)
   
   # remove punctuation
   # a list comprehension to remove punctuations
   dataset['comment_text'] = [re.sub('[^\w\s]|(\n)',' ', i) for i in dataset['comment_text']]

   return dataset



def data_generator(dataset):
    """
   a function to return x and y
   this function takes in dataset as argument.
   where
   x = list of comments
   y = a one hot vector of the classes
   """
    # get clean data: applying the function clean_punctuation
    dataset = clean_punctuation(dataset)    

    # create list and array
    x = list(dataset['comment_text'])
    y = np.array(dataset.iloc[:,2:])
    
    return[x,y]

data_generator('/home/kayode/KAYODE/Toxic_Kaggle/tain.csv')