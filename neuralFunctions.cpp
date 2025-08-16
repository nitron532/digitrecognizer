#include "neuralFunctions.h"
//consider passing things by reference for efficiency ( sigmoid functions)
auto sigmoid(Eigen::MatrixXd z){ //might wanna change the auto to the actual type?
    return 1.0/(1.0 + (-z).array().exp());
} 
auto sigmoidPrime(Eigen::MatrixXd z){
    return sigmoid(z)*(1-sigmoid(z));
}
Network::Network(std::vector<int>& lS){
    numLayers = layerSizes.size();
    for(size_t i = 1; i < numLayers; i++){
        biases.push_back(Eigen::VectorXd(lS[i]).Zero());
    }
    for(size_t i = 1; i < numLayers-1; i++){
        //uniform xavier initialization of weights tensor
        weights.push_back(Eigen::MatrixXd::Random(lS[i+1], lS[i]) * sqrt(6.0/(lS[i]+lS[i+1])));
    }
}

auto Network::feedForward(Eigen::VectorXd& prevLayerOutputs){
    for(size_t i = 0; i < biases.size(); i++){
        Eigen::VectorXd z = weights[i] * prevLayerOutputs + biases[i];
        prevLayerOutputs = sigmoid(z);
    }
    return prevLayerOutputs;
}

void Network::updateMiniBatch(std::vector<std::pair<double,int>> batch, double lR){
    
}

void Network::stochasticGradientDescent(std::vector<std::pair<double,int>> trainingData, size_t epochs, size_t miniBatchSize,double learningRate, std::vector<std::pair<double,int>> testData){
    size_t trainingDataSize = trainingData.size();
    std::random_device rd;
    std::mt19937 g(rd());
    for(size_t i = 0; i < epochs; i++){
        std::shuffle(trainingData.begin(), trainingData.end(),g);
        std::vector<std::vector<std::pair<double,int>>> miniBatches;
        for(size_t k = 0; k < trainingData.size(); k+=miniBatchSize){
            std::vector<std::pair<double,int>> batch(trainingData.begin()+k, trainingData.begin()+k+miniBatchSize);
            miniBatches.push_back(batch);
        }
        for(auto miniBatch : miniBatches){
            updateMiniBatch(miniBatch,learningRate);
        }
        // if(testData.size() != 0){
        //     std::cout<< "Epoch: " << i+1 << evaluate(test)
        // }
    }
}



