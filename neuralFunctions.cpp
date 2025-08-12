#include "neuralFunctions.h"

Network::Network(std::vector<int>& lS){
    std::random_device rd;
    std::mt19937 gen(rd());
    numLayers = layerSizes.size();
    for(size_t i = 1; i < numLayers; i++){
        Eigen::VectorXd iThLayerBiases(lS[i]);
        for(size_t j = 0; j < lS[i]; j++){
            iThLayerBiases(lS[i]) = rand() % lS[i]+1;
        }
        biases.push_back(iThLayerBiases);
    }
    for(size_t i = 1; i < numLayers-1; i++){
        std::uniform_int_distribution<> distr(lS[i+1], lS[i]);
        
    }
}

auto sigmoid(Eigen::MatrixXd z){ //might wanna change the auto to the actual type?
    return 1.0/(1.0 + (-z).exp());
} 
auto sigmoidPrime(Eigen::MatrixXd z){
    return sigmoid(z)*(1-sigmoid(z));
}
