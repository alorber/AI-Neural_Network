//
//  main.cpp
//  AI Neural Network Project
//
//  Created by Andrew Lorber on 11/23/19.
//  Copyright © 2019 Andrew Lorber. All rights reserved.
//

#include <iostream>
#include <vector>
#include <string>
#include <fstream>

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
    // Initialize Neural Network
    NeuralNetwork(string inputFile){
        
    }
    void learn();
    void test();
    void printWeights();
};

int main(int argc, const char * argv[]) {
    
    return 0;
}
