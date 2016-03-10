/*
 * Created by Paulius Sukys on 28.02.16.
 * Adapted from Dave Miller NN tutorial for C++ - http://www.millermattson.com/dave/?p=54
 */

#ifndef MYNN_NET_H
#define MYNN_NET_H
#include <vector>
#include <cassert>
#include <cmath>
#include "Layer.h"
#include "Neuron.h"


class Net {
public:
    /**
     * Constructs initial network
     * @param topology      object that defines parameters for a network
     * Topology structure:
     * integer array. Length of array - amount of arrays; entry integer - amount of neurons in layer
     */
    Net(const std::vector<unsigned> &topology);

    /**
     * Feeds values into the network as input
     * @param inputVals     input values
     */
    void feedForward(const std::vector<double> &inputVals);

    /**
     * Training function to show what the values really were:
     * back propagation
     * @param targetVals    result values
     */
    void backPropagate(const std::vector<double> &targetVals);

    /**
     * Reads output layer values.
     * Does not modify the network, thus const
     */
    void getResults(std::vector<double> &resultVals) const;

private:
    std::vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
    double m_error;
};


#endif //MYNN_NET_H
