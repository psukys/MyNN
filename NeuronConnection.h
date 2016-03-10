/*
 * Created by Paulius Sukys on 28.02.16.
 * Adapted from Dave Miller NN tutorial for C++ - http://www.millermattson.com/dave/?p=54
 */

#ifndef MYNN_CONNECTION_H
#define MYNN_CONNECTION_H
#include <cstdlib>

/**
 * Neuron connection class which consists of weights and last change in weight
 */
class NeuronConnection {
public:
    /**
     * Constructor: initiates a random weight in the range [0..1]
     */
    NeuronConnection(const double e, const double a);

    /**
     * Returns weight, const since this should not modify object
     */
    double getWeight(void) const;

    /**
     * Adjusts weight with given gradient
     * gradient     gradient value from next layer's neuron
     */
    void adjustWeight(const double gradient, const double output);

private:
    double weight;
    double deltaWeight;
    double eta;
    double alpha;
    double generateWeight();
};


#endif //MYNN_CONNECTION_H
