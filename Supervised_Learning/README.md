# Supervised Learning
Supervised learning is used whenever we want to predict a certain outcome from a given input, and we have examples of input/output pairs. We build a machine learning model from these input/output pairs, which comprise our training set. Our goal is to make accurate predictions for new, never-before-seen data. 

## Classification and Regression
There are two major types of supervised machine learning problems, called **classification** and **regression**.
-  In *classification*, the goal is to predict a class label, which is a choice from a predefined list of possibilities. 

    Classification is sometimes separated into *binary classification*, which is the special case of distinguishing between exactly two classes, and *multiclass  classification*, which is classification between more than two classes. You can think of binary classification as trying to answer a yes/no question. Classifying emails as either spam or not spam is an example of a binary classification problem.

- For *regression* tasks, the goal is to predict a continuous number, or a floating-point number in programming terms (or real number in mathematical terms).

An easy way to distinguish between classification and regression tasks is to ask
whether there is some kind of continuity in the output. If there is continuity between possible outcomes, then the problem is a regression problem.

## Generalization, Overfitting, and Underfitting

### Generalization
In supervised learning, we want to build a model on the training data and then be
able to make accurate predictions on new, unseen data that has the same characteristics as the training set that we used. If a model is able to make accurate predictions on unseen data, we say it is able to *generalize* from the training set to the test set. We want to build a model that is able to generalize as accurately as possible.

### Overfitting
Usually we build a model in such a way that it can make accurate predictions on the training set. If the training and test sets have enough in common, we expect the model to also be accurate on the test set. However, there are some cases where this can go wrong. For example, if we allow ourselves to build very *complex* models, we can always be as accurate as we like on the training set.

Building a model that is too complex for the amount of information we have is called *overfitting*. Overfitting  occurs when you fit a model too closely to the particularities of the training set and  obtain a model that works well on the training set but is not able to generalize to new data. 

### Underfitting
On the other hand, if your model is too simple then you might not be able to capture all the aspects of and variability in the data, and your model will do badly even on the training set. Choosing too simple a model is called *underfitting*.

## 
The more complex we allow our model to be, the better we will be able to predict on the training data. However, if our model becomes too complex, we start focusing too much on each individual data point in our training set, and the model will not generalize well to new data.

## Relation of Model Complexity to Dataset Size
The larger variety of data points your data set contains, the more complex a model you can use without overfitting. Usually, collecting more data points will yield more variety, so larger datasets allow building more complex models.