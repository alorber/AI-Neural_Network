# AI-Neural_Network
###### This is a neural network for project 2 of my artificial intelligence course.

The program allows for the training and testing of a neural network containing one hidden layer. The expected outputs 
for training and testing examples will always be 1 or 0, respectively indicating inclusion or exclusion from some Boolean 
class.

When the neural network is trained, the program will prompt the user for for the names of three text files 
representing the initial neural network, a training set, and an output file; one positive integer representing the 
number of epochs; and one floating-point value representing the learning rate. The first text file will specify the 
number of input nodes, hidden nodes, and output nodes. The training is done using back-propagation.

When the neural network is tested, the program will prompt the user for the names of three text files representing a 
trained neural network, a testing set, and an output file. Four metrics are calculated for each Boolean output class: 
Overall Accuracy, Precision, Recall, and F1. These metrics are also calculated for the entire data-set using mico-averaging 
and macro-averaging. 

Additionally, each student was required to create or find an interesting dataset that can be used to train and test neural 
networks adhering to the specifications of this assignment.

My custom data-set is a modified car evaluation database from the UCI Machine Learning Repository [1]. The car evaluation 
database describes the acceptability of a vehicle given certain attributes. The input attributes and their original possible 
values were:

| Attribute         | Possible Values        |
| ----------------- | ---------------------- |
| Buy Price         | v-high, high, med, low |
| Maintenance Price | v-high, high, med, low |
| Number of Doors   | 2, 3, 4, 5-more        |
| Max Capacity      | 2, 4, more             |
| Trunk Size        | Small, med, big        |
| Safety            | Low, med, high         |

In order to use this dataset I wrote a program that converted each string value into a double value, representing the 
option number normalized by the maximum number of options for the given input attribute. The updated possible values are:

| Attribute         | Possible Values        |
| ----------------- | ---------------------- |
| Buy Price         | 0.25, 0.50, 0.75, 1.00 |
| Maintenance Price | 0.25, 0.50, 0.75, 1.00 |
| Number of Doors   | 0.25, 0.50, 0.75, 1.00 |
| Max Capacity      | 0.333, 0.666, 0.999    |
| Trunk Size        | 0.333, 0.666, 0.999    |
| Safety            | 0.333, 0.666, 0.999    |

The output of the original dataset was a string representing the acceptability of the car. The possible values were “unacc”, 
“acc”, “good”, and “v-good”. I modified the output of the dataset to be four Boolean outputs, each representing one of the 
original choices.

The initial weights of the neural network were randomly generated values between 0 and 1. I wrote a small program that would 
create the initial weights text file given the number of nodes in each layer. The parameter combination that worked the best 
of those tested was 5 hidden nodes, 200 epochs, and a 0.2 learning rate.

The original dataset contained 1728 examples. The distribution of these examples was:

| Class    | Amount | Percentage of Total |
| -------- | ------ | ------------------- |
| “unacc”  | 1210   | 70.023 %            |
| “acc”    | 384    | 22.222 %            |
| “good”   | 69     | 3.993 %             |
| “v-good” | 65     | 3.762 %             |

The dataset was randomly split between the training and testing file, with a corresponding 3:1 size ratio.

________________________________
[1] https://archive.ics.uci.edu/ml/datasets/car+evaluation
