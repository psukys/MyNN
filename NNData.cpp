/*
 * Created by Paulius Sukys on 28.02.16.
 * Adapted from Dave Miller NN tutorial for C++ - http://www.millermattson.com/dave/?p=54
 */

#include <sstream>
#include "NNData.h"

void NNData::getTopology(std::vector<unsigned> &topology) {
    std::string line;
    std::string label;

    getline(m_nnDataFile, line);
    std::stringstream ss(line);
    ss >> label;
    if (this->isEof() || label.compare("topology:") != 0) {
        abort();
    }

    while (!ss.eof()) {
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }
}

NNData::NNData(const std::string filename) {
    m_nnDataFile.open(filename.c_str());
}

unsigned long NNData::getInputVals(std::vector<double> &inputVals) {
    inputVals.clear();

    std::string line;
    getline(m_nnDataFile, line);
    std::stringstream ss(line);

    std::string label;
    ss >> label;
    if (label.compare("in:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            inputVals.push_back(oneValue);
        }
    }
    return inputVals.size();
}

unsigned long NNData::getTargetVals(std::vector<double> &targetVals) {
    targetVals.clear();

    std::string line;
    getline(m_nnDataFile, line);
    std::stringstream ss(line);

    std::string label;
    ss >> label;
    if (label.compare("out:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            targetVals.push_back(oneValue);
        }
    }
    return targetVals.size();
}

bool NNData::isEof() {
    return m_nnDataFile.eof();
}