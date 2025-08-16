#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <algorithm>
#include <random>
#include <unordered_map>
using Eigen::MatrixXd;
using Eigen::VectorXd;
class Network{
    private:
        size_t numLayers;
        std::vector<int> layerSizes;
        std::vector<Eigen::VectorXd> biases;
        std::vector<Eigen::MatrixXd> weights;
        void updateMiniBatch(const std::vector<std::pair<Eigen::VectorXd,Eigen::VectorXd>> batch, double lR);
        std::pair<std::vector<Eigen::VectorXd>,std::vector<Eigen::VectorXd>> backpropagation(Eigen::VectorXd x, Eigen::VectorXd y);
        Eigen::VectorXd costDerivative(Eigen::VectorXd outputActivations, Eigen::VectorXd y);
    public:
        Network(std::vector<int>& lS);
        auto feedForward(Eigen::VectorXd previousLayerOutput);
        void stochasticGradientDescent(std::vector<std::pair<Eigen::VectorXd,Eigen::VectorXd>> trainingData, size_t epochs, size_t miniBatchSize, double learningRate,std::vector<std::pair<Eigen::VectorXd,Eigen::VectorXd>> testData);
};