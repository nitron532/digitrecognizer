#include "neuralFunctions.h"
//consider passing things by reference for efficiency ( sigmoid functions)
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
        prevLayerOutputs = (weights[i] * prevLayerOutputs + biases[i]).unaryExpr(&sigmoid);
    }
    return prevLayerOutputs;
}



auto sigmoid(Eigen::MatrixXd z){ //might wanna change the auto to the actual type?
    return 1.0/(1.0 + (-z).exp());
} 
auto sigmoidPrime(Eigen::MatrixXd z){
    return sigmoid(z)*(1-sigmoid(z));
}