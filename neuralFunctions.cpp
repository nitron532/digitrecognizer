#include "neuralFunctions.h"

double reLu(double x){
    return std::max(0.0,x);
}
double reLuPrime(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

Eigen::VectorXd softMax(const Eigen::VectorXd& activationVector) {
    double maxCoeff = activationVector.maxCoeff();
    Eigen::VectorXd expShifted = (activationVector.array() - maxCoeff).exp();
    double denominator = expShifted.sum();
    return expShifted / denominator;
}

Network::Network(std::vector<size_t> sizes){
    std::random_device rd;
    std::mt19937 gen(rd());
    numLayers = sizes.size();
    for(size_t i = 0; i < sizes.size()-1; i++){
        std::normal_distribution<double> he(0,sqrt(2/sizes[i]));
        weights.push_back(Eigen::MatrixXd::Constant(sizes[i+1],sizes[i],1) * he(rd));
        biases.push_back(Eigen::VectorXd::Zero(sizes[i+1]));
    }
}

std::vector<std::vector<Eigen::VectorXd>> Network::feedForwardOneBatch(const stdVecStdPairEigVec& trainingData, size_t& marker, size_t miniBatchSize){
    size_t until = marker+miniBatchSize;
    if(until >= trainingData.size()){ //prevent until going over trainingData size
        until = trainingData.size();
    }
    std::vector<std::vector<Eigen::VectorXd>> batchActivationVectors;
    for(size_t i = marker; i < until; i++){ //iterate thru original trainingData in increments of miniBatch size
        Eigen::VectorXd activation = trainingData[i].first;
        std::vector<Eigen::VectorXd> activationsStdVector = {activation};
        for(size_t j = 0; j < numLayers-2;j++){ //feedsforward one image thru ReLu
            activation = weights[j] * activation + biases[j];
            activation = activation.unaryExpr(&reLu);
        }
        activationsStdVector.push_back(softMax(activationsStdVector[activationsStdVector.size()-1])); //softmax final activation
        batchActivationVectors.push_back(activationsStdVector);
    }
    return batchActivationVectors;
}

void Network::sgdTrain(stdVecStdPairEigVec& trainingData, size_t miniBatches, size_t epochs, double learningRate){
    auto rng = std::default_random_engine {};
    for(size_t i = 0; i < epochs; i++){
        std::shuffle(trainingData.begin(),trainingData.end(), rng);
        size_t startAt = 0;
        for(size_t j = 0; j < miniBatches; j++){
            std::vector<std::vector<Eigen::VectorXd>> batchActivations = feedForwardOneBatch(trainingData, startAt, miniBatches);
            //backprop here
        }
    }
}
