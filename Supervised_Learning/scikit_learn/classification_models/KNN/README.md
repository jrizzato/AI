# k-Nearest Neighbors
The k-NN algorithm is arguably the simplest machine learning algorithm. Building the model consists only of storing the training dataset. To make a prediction for a new data point, the algorithm finds the closest data points in the training dataset—its “nearest neighbors".

Instead of considering only the closest neighbor, we can also consider an arbitrary  number, k, of neighbors. This is where the name of the k-nearest neighbors algorithm  comes from. When considering more than one neighbor, we use voting to assign a  label. This means that for each test point, we count how many neighbors belong to  class 0 and how many neighbors belong to class 1. We then assign the class that is  more frequent: in other words, the majority class among the k-nearest neighbors. 

This method can be applied to datasets with any number of classes. For more classes, we count how many neighbors belong to each class and again predict the most common class.

## Strengths, weaknesses, and parameters  
In principle, there are two important parameters to the KNeighbors classifier: the number of neighbors and how you measure distance between data points. In practice, using a small number of neighbors like three or five often works well, but you should certainly adjust this parameter. Choosing the right distance measure, by default, Euclidean distance is used, which works well in many settings.  

One of the strengths of k-NN is that the model is very easy to understand, and often gives reasonable performance without a lot of adjustments. Using this algorithm is a good baseline method to try before considering more advanced techniques.  
Building the nearest neighbors model is usually very fast, but when your training set is very large (either in number of features or in number of samples) prediction can be slow.  
 
This approach often does not perform well on datasets with many features (hundreds or more), and it does particularly badly with datasets where most features are 0 most of the time (so-called sparse datasets).  
