/*
 * Created by Paulius Sukys on 28.02.16.
 * Adapted from Dave Miller NN tutorial for C++ - http://www.millermattson.com/dave/?p=54
 */

#include <iostream>
#include <vector>
#include "Net.h"
#include "NNData.h"

void printDoubleVector(std::vector<double> vec) {
    for (unsigned i = 0; i < vec.size(); i++) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    NNData trainData("/tmp/trainingData.txt");
    std::vector<unsigned> topology;
    trainData.getTopology(topology);

    Net myNN(topology);

    std::vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;

    while (!trainData.isEof()) {
        ++trainingPass;
        std::cout << std::endl << "Pass " << trainingPass << std::endl;
        if (trainData.getInputVals(inputVals) != topology[0]) {
            break;
        }

        myNN.feedForward(inputVals);
        std::cout << "Input:" << std::endl;
        printDoubleVector(inputVals);

        myNN.getResults(resultVals);
        std::cout << "Results:" << std::endl;
        printDoubleVector(resultVals);

        trainData.getTargetVals(targetVals);
        std::cout << "Target:" << std::endl;
        printDoubleVector(targetVals);

        myNN.backPropagate(targetVals);
    }
    std::cout << std::endl << "Fin." << std::endl;
    return 0;
}