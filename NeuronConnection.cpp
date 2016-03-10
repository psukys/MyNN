/*
 * Created by Paulius Sukys on 28.02.16.
 * Adapted from Dave Miller NN tutorial for C++ - http://www.millermattson.com/dave/?p=54
 */

#include "NeuronConnection.h"

NeuronConnection::NeuronConnection(const double e, const double a) {
    eta = e;
    alpha = a;
    weight = generateWeight();
}

double NeuronConnection::generateWeight() {
    return rand() / double(RAND_MAX);
}

double NeuronConnection::getWeight() const {
    return weight;
}

void NeuronConnection::adjustWeight(const double gradient, const double output) {
    deltaWeight = eta * output * gradient + alpha * deltaWeight;
    weight += deltaWeight;
}
