#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <algorithm>
#include <random>
#include <ctime>
#include <fstream>
typedef std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> imagesInputAndValue;
class Network{
    private:
        size_t inputSize = 1;
        size_t numLayers;
        std::vector<Eigen::MatrixXd> weights;
        std::vector<Eigen::VectorXd> biases;
        std::vector<Eigen::MatrixXd> feedForwardOneBatch(const Eigen::MatrixXd& batch, std::vector<Eigen::MatrixXd>& zs);
        void backPropagation(const std::vector<Eigen::MatrixXd>& batchActivations, const std::vector<Eigen::MatrixXd>& zs, const Eigen::MatrixXd& oneHots, size_t thisBatchSize, double learningRate, double reg);
    public:
        Network(std::vector<size_t> sizes);
        void sgdTrain(imagesInputAndValue& trainingData, size_t miniBatchSize, size_t epochs, double learningRate,const imagesInputAndValue& testingData, double reg);
        void testNetwork(const imagesInputAndValue& testingData);
};