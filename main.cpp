//
//  main.cpp
//  AI Neural Network Project
//
//  Created by Andrew Lorber on 11/23/19.
//  Copyright © 2019 Andrew Lorber. All rights reserved.
//

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <math.h>

using namespace std;

// Neural Network Class
class NeuralNetwork {
private:
    class Node {
    public:
        double input;
        double activation;
        double biasWeight;
        // Weights of edges pointing from previous layer to current node
        vector<double> inputWeights;
    };
    // Layers
    int numInputNodes;
    vector<Node> inputLayer;
    int numHiddenNodes;
    vector<Node> hiddenLayer;
    int numOutputNodes;
    vector<Node> outputLayer;
    
public:
    NeuralNetwork(string inputFile);
    void learn();
    void test();
    void printWeights(string outputFile);
    // Sigmoid activation function
    double sigmoid(double value){
        return (1 / (1 + exp(value * (-1))));
    }
    // Derivative of the sigmoid activation function
    double sigmoidPrime(double value){
        return (sigmoid(value) * (1 - sigmoid(value)));
    }
};

// Initialize Neural Network
NeuralNetwork::NeuralNetwork(string inputFile){
    // Opens initialization file
    ifstream fin;
    fin.open(inputFile);
    fin >> numInputNodes >> numHiddenNodes >> numOutputNodes;
    
    // Initializes layers
    Node emptyNode;
    double weight;
    
    // Input Layer
    for(int i = 0; i < numInputNodes; i++){
        inputLayer.push_back(emptyNode);
    }
    // Hidden Layer
    for(int i = 0; i < numHiddenNodes; i++){
        hiddenLayer.push_back(emptyNode);
        
        // Adds weights to node
        fin >> hiddenLayer.at(i).biasWeight;
        
        for(int j = 0; j < numInputNodes; j++){
            fin >> weight;
            hiddenLayer.at(i).inputWeights.push_back(weight);
        }
    }
    // Output Layer
    for(int i = 0; i < numOutputNodes; i++){
        outputLayer.push_back(emptyNode);
        
        // Adds weights to node
        fin >> outputLayer.at(i).biasWeight;
        for(int j = 0; j < numHiddenNodes; j++){
            fin >> weight;
            outputLayer.at(i).inputWeights.push_back(weight);
        }
    }
    fin.close();
}

// Trains the Neural Network
void NeuralNetwork::learn(){
    string trainingFile;
    string outFile;
    int numEpochs;
    double learningRate;
    
    cout << "Please enter the name of the file containing the training set.\n";
    cin >> trainingFile;
    ifstream fin;
    fin.open(trainingFile);
    // Checks for valid file name
    while(!fin.is_open()){
        cout << "The file you have entered cannot be found. Please enter a new file name.\n";
        cin >> trainingFile;
        fin.open(trainingFile);
    }
    
    cout << "Please enter the name of the output file.\n";
    cin >> outFile;
    
    cout << "Please enter the number of number of epochs.\n";
    cin >> numEpochs;
    
    cout << "Please enter the learning rate.\n";
    cin >> learningRate;
    
    // Loads training examples
    int numExamples;
    fin >> numExamples;
        // Skips to next line
    fin.ignore(256, '\n');
    
        // Each example contains a vector of inputs and a vector of outputs
    vector<vector<vector<double>>> examples;
    vector<vector<double>> example;
    vector<double> eInputs;
    vector<double> eOutputs;
    double value;
    
    for(int i = 0; i < numExamples; i++){
        example.clear();
        eInputs.clear();
        eOutputs.clear();
        
        for(int j = 0; j < numInputNodes; j++){
            fin >> value;
            eInputs.push_back(value);
        }
        for(int j = 0; j < numOutputNodes; j++){
            fin >> value;
            eOutputs.push_back(value);
        }
        example.push_back(eInputs);
        example.push_back(eOutputs);
        examples.push_back(example);
    }
    fin.close();
    
    for(int i = 0; i < numEpochs; i++){
        for(int j = 0; j < numExamples; j++){
            // Initializes inputs
            for(int k = 0; k < numInputNodes; k++){
                inputLayer.at(k).input = inputLayer.at(k).activation = examples.at(j).at(0).at(k);
            }
            
            // Propagate forward to hidden layer
            double sum;
            for(int k = 0; k < numHiddenNodes; k++){
                sum = hiddenLayer.at(k).biasWeight * (-1);
                for(int l = 0; l < numInputNodes; l++){
                    sum += hiddenLayer.at(k).inputWeights.at(l) * inputLayer.at(l).activation;
                }
                hiddenLayer.at(k).input = sum;
                hiddenLayer.at(k).activation = sigmoid(sum);
            }
            // Propagate forward to output layer
            for(int k = 0; k < numOutputNodes; k++){
                sum = outputLayer.at(k).biasWeight * (-1);
                for(int l = 0; l < numHiddenNodes; l++){
                    sum += outputLayer.at(k).inputWeights.at(l) * hiddenLayer.at(l).activation;
                }
                outputLayer.at(k).input = sum;
                outputLayer.at(k).activation = sigmoid(sum);
            }
            
            vector<double> outputErrors;
            vector<double> hiddenErrors;
            // Calculate error at output layer
            for(int k = 0; k < numOutputNodes; k++){
                value = sigmoidPrime(outputLayer.at(k).input)
                    * (examples.at(j).at(1).at(k) - outputLayer.at(k).activation);
                outputErrors.push_back(value);
            }
            // Propagate backward to hidden layer
            for(int k = 0; k < numHiddenNodes; k++){
                sum = 0;
                for(int l = 0; l < numOutputNodes; l++){
                    sum += outputLayer.at(l).inputWeights.at(k) * outputErrors.at(l);
                }
                value = sigmoidPrime(hiddenLayer.at(k).input) * sum;
                hiddenErrors.push_back(value);
            }
            
            // Update weights according to deltas
                // Input layer -> hidden layer
            for(int k = 0; k < numHiddenNodes; k++){
                for(int l = 0; l < numInputNodes; l++){
                    hiddenLayer.at(k).inputWeights.at(l) += (learningRate * inputLayer.at(l).activation * hiddenErrors.at(k));
                }
                hiddenLayer.at(k).biasWeight += (learningRate * (-1) * hiddenErrors.at(k));
            }
                // Hidden layer -> output layer
            for(int k = 0; k < numOutputNodes; k++){
                for(int l = 0; l < numHiddenNodes; l++){
                    outputLayer.at(k).inputWeights.at(l) += (learningRate * hiddenLayer.at(l).activation * outputErrors.at(k));
                }
                outputLayer.at(k).biasWeight += (learningRate * (-1) * outputErrors.at(k));
            }
        }
    }
    printWeights(outFile);
}

// Tests the Neural Network
void NeuralNetwork::test(){
    string testFile;
    string outFile;
    
    cout << "Please enter the name of the file containing the testing set.\n";
    cin >> testFile;
    ifstream fin;
    fin.open(testFile);
    // Checks for valid file name
    while(!fin.is_open()){
        cout << "The file you have entered cannot be found. Please enter a new file name.\n";
        cin >> testFile;
        fin.open(testFile);
    }
    
    cout << "Please enter the name of the output file.\n";
    cin >> outFile;
    
    // Loads testing examples
    int numExamples;
    fin >> numExamples;
        // Skips to next line
    fin.ignore(256, '\n');
    
        // Each example contains a vector of inputs and a vector of outputs
    vector<vector<vector<double>>> examples;
    vector<vector<double>> example;
    vector<double> eInputs;
    vector<double> eOutputs;
    double value;
    
    for(int i = 0; i < numExamples; i++){
        example.clear();
        eInputs.clear();
        eOutputs.clear();
        
        for(int j = 0; j < numInputNodes; j++){
            fin >> value;
            eInputs.push_back(value);
        }
        for(int j = 0; j < numOutputNodes; j++){
            fin >> value;
            eOutputs.push_back(value);
        }
        example.push_back(eInputs);
        example.push_back(eOutputs);
        examples.push_back(example);
    }
    fin.close();
    vector<vector<int>> outputs;
    vector<int> tOutputs;
    for(int i = 0; i < numExamples; i++){
        tOutputs.clear();
        // Initializes inputs
        for(int j = 0; j < numInputNodes; j++){
            inputLayer.at(j).input = inputLayer.at(j).activation = examples.at(i).at(0).at(j);
        }
        
        // Propagate forward to hidden layer
        double sum;
        for(int j = 0; j < numHiddenNodes; j++){
            sum = hiddenLayer.at(j).biasWeight * (-1);
            for(int k = 0; k < numInputNodes; k++){
                sum += hiddenLayer.at(j).inputWeights.at(k) * inputLayer.at(k).activation;
            }
            hiddenLayer.at(j).input = sum;
            hiddenLayer.at(j).activation = sigmoid(sum);
        }
        // Propagate forward to output layer
        for(int j = 0; j < numOutputNodes; j++){
            sum = outputLayer.at(j).biasWeight * (-1);
            for(int k = 0; k < numHiddenNodes; k++){
                sum += outputLayer.at(j).inputWeights.at(k) * hiddenLayer.at(k).activation;
            }
            outputLayer.at(j).input = sum;
            outputLayer.at(j).activation = sigmoid(sum);
            if(outputLayer.at(j).activation >= 0.5){
                tOutputs.push_back(1);
            } else {
                tOutputs.push_back(0);
            }
        }
        outputs.push_back(tOutputs);
    }
}

// Prints the weights
void NeuralNetwork::printWeights(string outputFile){
    ofstream fout;
    fout.open(outputFile);
    fout << numInputNodes << " " << numHiddenNodes << " " << numOutputNodes << '\n';
    // Prints all weights
    // Input layer -> hidden layer
    for(int i = 0; i < numHiddenNodes; i++){
        fout << fixed << setprecision(3) << hiddenLayer.at(i).biasWeight;
        for(int j = 0; j < numInputNodes; j++){
            fout << " " << fixed << setprecision(3) << hiddenLayer.at(i).inputWeights.at(j);
        }
        fout << '\n';
    }
    // Hidden layer -> output layer
    for(int i = 0; i < numOutputNodes; i++){
        fout << fixed << setprecision(3) << outputLayer.at(i).biasWeight;
        for(int j = 0; j < numHiddenNodes; j++){
            fout << " " << fixed << setprecision(3) << outputLayer.at(i).inputWeights.at(j);
        }
        fout << '\n';
    }
    fout.close();
}

int main(int argc, const char * argv[]) {
    string inputFile;
    cout << "Please enter the name of the file containing the starting weights.\n";
    cin >> inputFile;
    
    ifstream fin;
    fin.open(inputFile);
    // Checks for valid file name
    while(!fin.is_open()){
        cout << "The file you have entered cannot be found. Please enter a new file name.\n";
        cin >> inputFile;
        fin.open(inputFile);
    }
    fin.close();
    
    NeuralNetwork net = NeuralNetwork(inputFile);
    net.test();
    return 0;
}
