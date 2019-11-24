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
        // Opens initialization file
        ifstream fin;
        fin.open(inputFile);
        fin >> numInputNodes >> numHiddenNodes >> numOutputNodes;
        
        // Initializes layers
        Node emptyNode;
        float weight;
        
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
    }
    void learn();
    void test();
    void printWeights();
};

int main(int argc, const char * argv[]) {
    NeuralNetwork net = NeuralNetwork("init.txt");
    return 0;
}
