#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::VectorXd;
class Network{
    private:
        size_t numLayers;
        std::vector<int> layerSizes;
        std::vector<Eigen::VectorXd> biases;
        std::vector<Eigen::MatrixXd> weights;
    public:
        Network(std::vector<int>& lS);
        auto feedForward(Eigen::VectorXd& previousLayerOutput);
};