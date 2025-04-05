# k-Nearest Neighbors
The k-NN algorithm is arguably the simplest machine learning algorithm. Building the model consists only of storing the training dataset. To make a prediction for a new data point, the algorithm finds the closest data points in the training dataset—its “nearest neighbors".

Instead of considering only the closest neighbor, we can also consider an arbitrary  number, k, of neighbors. This is where the name of the k-nearest neighbors algorithm  comes from. When considering more than one neighbor, we use voting to assign a  label. This means that for each test point, we count how many neighbors belong to  class 0 and how many neighbors belong to class 1. We then assign the class that is  more frequent: in other words, the majority class among the k-nearest neighbors. 

This method can be applied to datasets with any number of classes. For more classes, we count how many neighbors belong to each class and again predict the most common class.