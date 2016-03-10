/*
 * Created by Paulius Sukys on 28.02.16.
 * Adapted from Dave Miller NN tutorial for C++ - http://www.millermattson.com/dave/?p=54
 */

#ifndef MYNN_NEURON_H
#define MYNN_NEURON_H
#include <vector>
#include "NeuronConnection.h"

/*
 * Regarding layer usage: couldn't figure out a hierarchy which would satisfy without importing layer header file to neuron.h
 * Thus the solution (probably bad, since the data type is sort of hardcoded) is to specify what is Neuron expecting
 */

/**
 * Atomic representation of a single neuron which is capable of computing output, given input, and likewise
 * do back-propagation through its activation function derivative
 * eta          Overall net training rate
 * alpha        Last weight change multiplier (momentum)
 * m_outputVal  Neuron's output
 * m_index      Index given to neuron (layer-wise)
 * m_gradient   Neuron's gradient
 * Usage example: see class Net
 */
class Neuron {
public:
    /**
     * Constructor: sets Neuron layer-wise index and supplies information
     * about how many neurons are in the next layer
     * numOutputs   amount of neurons in next layer
     * index        layer-wise index
     */
    Neuron(unsigned numOutputs, unsigned index);

    /**
     * Sums previous layer's outputs
     * Includes bias node from previous layer
     * prevLayer    previous layer
     */
    void feedForward(std::vector<Neuron> &prevLayer);

    /**
     * Sets Neuron's output value
     * val      value to set
     */
    void setOutputVal(double val);

    /**
     * Gets Neuron's output value. Const because does not modify anything
     */
    double getOutputVal(void) const;

    /**
     * Calculates output layer gradients for every neuron connection
     * targetVal    actual value from training samples
     */
    void calculateOutputGradients(double targetVal);

    /**
     * Calculates hidden layer output gradients for every neuron connection
     * nextLayer    next layer
     */
    void calculateHiddenGradients(const std::vector<Neuron> &nextLayer);

    /**
     * Update input weights for a previous layer
     * prevLayer    previous layer
     */
    void updateInputWeights(std::vector<Neuron> &prevLayer);

    /**
     * Updates output to specific neuron weight, this is usually called from next layer to previous layer neurons
     * index        index of output weight, which also corresponds to self-index of neuron from next layer
     * gradient     gradient of next layer neuron
     */
    void updateInputWeight(const int index, const double gradient);

private:
    double m_outputVal;
    unsigned m_index;
    double m_gradient;
    std::vector<NeuronConnection> m_connections;//m_outputWeights;

    /**
     * Activation function - definition of neuron output given specific input.
     * Currently the activation function is hyperbolic tangent function
     * sum   sum of inputs
     */
    static double activationFunction(double sum);

    /**
     * Activation function derivative - used for back propagation
     * Currently a derivative of hyperbolic tangent function: 1 - tanh^2
     * sum   sum of inputs
     */
    static double activationFunctionDerivative(double sum);

    /**
     * Sum of derivative weights
     * layer     list of neurons to sum up
     */
    double sumWeightsOfDerivatives(const std::vector<Neuron> &layer);
};

#endif //MYNN_NEURON_H
