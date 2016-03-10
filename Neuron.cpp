/*
 * Created by Paulius Sukys on 28.02.16.
 * Adapted from Dave Miller NN tutorial for C++ - http://www.millermattson.com/dave/?p=54
 */

#include <cmath>
#include "Neuron.h"

Neuron::Neuron(unsigned numOutput, unsigned index) {
    m_index = index;

    // Create informational (weight wise) connection nodes for every next layer neuron
    for (unsigned conn = 0; conn < numOutput; ++conn) {
        m_connections.push_back(NeuronConnection(/*eta=*/0.15, /*alpha=*/0.5));
    }
}

void Neuron::setOutputVal(double val) { m_outputVal = val; }

double Neuron::getOutputVal() const { return m_outputVal; }

void Neuron::feedForward(std::vector<Neuron> &prevLayer) {
    // Scaled sum of previous layer neuron outputs
    double sum = 0.0;

    // Sum the scaled outputs of every previous layer's neuron
    for (unsigned neuronNum = 0; neuronNum < prevLayer.size(); ++neuronNum) {
        sum += prevLayer[neuronNum].getOutputVal() * prevLayer[neuronNum].m_connections[m_index].getWeight();
    }

    // Run activation function on the scaled sum of output
    m_outputVal = Neuron::activationFunction(sum);
}

double Neuron::activationFunction(double sum) {
    // tanh - [-1.0,...,1.0]
    return tanh(sum);
}

double Neuron::activationFunctionDerivative(double sum) {
    // NOTE: approximation of tanh derivative : 1 - sum^2
    return 1 - tanh(sum) * tanh(sum);
}

void Neuron::calculateOutputGradients(double targetVal) {
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::activationFunctionDerivative(m_outputVal);
}

void Neuron::calculateHiddenGradients(const std::vector<Neuron> &nextLayer) {
    double dow = sumWeightsOfDerivatives(nextLayer);
    m_gradient = dow * Neuron::activationFunctionDerivative(m_outputVal);
}

double Neuron::sumWeightsOfDerivatives(const std::vector<Neuron> &layer) {
    double sum = 0.0;
    //sum up contributions of the errors for feeded nodes
    for (unsigned neuronNum = 0; neuronNum < layer.size() - 1; ++neuronNum) {
        sum += m_connections[neuronNum].getWeight() * layer[neuronNum].m_gradient;
    }
    return sum;
}

void Neuron::updateInputWeight(const int index, const double gradient) {
    m_connections[index].adjustWeight(gradient, m_outputVal);
}

void Neuron::updateInputWeights(std::vector<Neuron> &prevLayer) {
    // Weights to be updated are in connection object
    // in each neuron in preceding layer
    for (unsigned neuronNum = 0; neuronNum < prevLayer.size(); ++neuronNum) {
        prevLayer[neuronNum].updateInputWeight(m_index, m_gradient);
    }
}