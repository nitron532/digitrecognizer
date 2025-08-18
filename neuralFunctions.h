#define EIGEN_NO_DEBUG 0   // make sure debug is on
#define EIGEN_INITIALIZE_MATRICES_BY_NAN 1 
#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <fstream>
typedef std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> stdVectorPairEigVector;
using Eigen::MatrixXd;
using Eigen::VectorXd;
class Network{
    private:
        size_t numLayers;
        std::vector<size_t> layerSizes;
        std::vector<Eigen::VectorXd> biases;
        std::vector<Eigen::MatrixXd> weights;
        void updateMiniBatch(const stdVectorPairEigVector batch, double lR);
        std::pair<std::vector<Eigen::VectorXd>,std::vector<Eigen::MatrixXd>> backpropagation(Eigen::VectorXd x, Eigen::VectorXd y);
        Eigen::VectorXd costDerivative(Eigen::VectorXd outputActivations, Eigen::VectorXd y);
        std::pair<size_t,size_t> evaluate(stdVectorPairEigVector trainingData);
        Eigen::VectorXd feedForward(Eigen::VectorXd previousLayerOutput);
    public:
        Network(std::vector<size_t>& lS);
        void stochasticGradientDescent(stdVectorPairEigVector trainingData, size_t epochs, size_t miniBatchSize, double learningRate,stdVectorPairEigVector testData);
};