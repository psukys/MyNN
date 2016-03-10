/*
 * Created by Paulius Sukys on 28.02.16.
 * Adapted from Dave Miller NN tutorial for C++ - http://www.millermattson.com/dave/?p=54
 */

#include "Net.h"

Net::Net(const std::vector<unsigned> &topology) {
    unsigned long numLayers = topology.size();

    // Adding layers
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(Layer());
        unsigned numOutputs;
        if (layerNum == topology.size() - 1) {
            // output layer is the final layer
            numOutputs = 0;
        } else {
            // amount of elements in the next layer
            numOutputs = topology[layerNum + 1];
        }

        // Filling neurons
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
        }
        // Set bias neuron to constant 1.0 (nothing changes it)
        m_layers.back().back().setOutputVal(1.0);
    }
}

void Net::feedForward(const std::vector<double> &inputVals) {
    assert(inputVals.size() == m_layers[0].size() - 1); //subtract bias neuron
    // Assign input values to input neurons
    for (unsigned i = 0; i < inputVals.size(); ++i) {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    // Forward propagation, skip input layer
    for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
        Layer &prevLayer = m_layers[layerNum - 1];
        for (unsigned neuronNum = 0; neuronNum < m_layers[layerNum].size() - 1; ++neuronNum) {
            // Individual feedforward for each neuron
            m_layers[layerNum][neuronNum].feedForward(prevLayer);
        }
    }
}

void Net::backPropagate(const std::vector<double> &targetVals) {
    // net error (RMS)
    Layer &outputLayer = m_layers.back(); // readable
    m_error = 0.0;

    for (unsigned neuronNum = 0; neuronNum < outputLayer.size() - 1; ++neuronNum) {
        double delta = targetVals[neuronNum] - outputLayer[neuronNum].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1;
    m_error = sqrt(m_error);

    // output layer gradients
    for (unsigned neuronNum = 0; neuronNum < outputLayer.size() - 1; ++neuronNum) {
        outputLayer[neuronNum].calculateOutputGradients(targetVals[neuronNum]);
    }

    // hidden layer gradients
    for (unsigned long layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum - 1];
        for (unsigned neuronNum = 0; neuronNum < hiddenLayer.size(); ++neuronNum) {
            hiddenLayer[neuronNum].calculateHiddenGradients(nextLayer);
        }
    }

    // update connection weights
    for (unsigned long layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        for (unsigned neuronNum = 0; neuronNum < layer.size() - 1; ++neuronNum) {
            layer[neuronNum].updateInputWeights(prevLayer);
        }

    }
}

void Net::getResults(std::vector<double> &resultVals) const {
    resultVals.clear();
    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
        resultVals.push_back((m_layers.back()[n].getOutputVal()));
    }
}