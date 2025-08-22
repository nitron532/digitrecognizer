#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <algorithm>
#include <random>
#include <ctime>
#include <fstream>
typedef std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> stdVecStdPairEigVec;
class Network{
    private:
        size_t numLayers;
        std::vector<Eigen::MatrixXd> weights;
        std::vector<Eigen::VectorXd> biases;
        std::vector<std::vector<Eigen::VectorXd>> feedForwardOneBatch(const stdVecStdPairEigVec& trainingData, size_t& marker, size_t miniBatches);
    public:
        Network(std::vector<size_t> sizes);
        void sgdTrain(stdVecStdPairEigVec& trainingData, size_t miniBatches, size_t epochs, double learningRate);
        void testNetwork(const stdVecStdPairEigVec& testingData);
};