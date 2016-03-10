/*
 * Created by Paulius Sukys on 28.02.16.
 * Adapted from Dave Miller NN tutorial for C++ - http://www.millermattson.com/dave/?p=54
 */

#ifndef MYNN_NNDATA_H
#define MYNN_NNDATA_H

#include <iostream>
#include <vector>
#include <fstream>

class NNData {
public:
    NNData(const std::string filename);
    bool isEof(void);
    /**
     * Returns the topology for neural network initialization
     * Topology structure:
     * integer array. Length of array - amount of arrays; entry integer - amount of neurons in layer
     */
    void getTopology(std::vector<unsigned> &topology);
    unsigned long getInputVals(std::vector<double> &inputVals);
    unsigned long getTargetVals(std::vector<double> &targetVals);

private:
    std::ifstream m_nnDataFile;
};


#endif //MYNN_NNDATA_H
