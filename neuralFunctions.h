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
        void updateMiniBatch(std::vector<std::pair<double,int>> batch, double lR);
    public:
        Network(std::vector<int>& lS);
        auto feedForward(Eigen::VectorXd& previousLayerOutput);
        void stochasticGradientDescent(std::vector<std::pair<double,int>> trainingData, size_t epochs, size_t miniBatchSize, double learningRate,std::vector<std::pair<double,int>> testData);
};