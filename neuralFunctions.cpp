#include "neuralFunctions.h"
//consider passing things by reference for efficiency ( sigmoid functions)
Eigen::MatrixXd sigmoid(Eigen::MatrixXd z){ //might wanna change the auto to the actual type?
    return 1.0/(1.0 + (-z).array().exp());
} 
Eigen::MatrixXd sigmoidPrime(Eigen::MatrixXd z){
    return (sigmoid(z).array()*(1-sigmoid(z).array())).matrix();
}

Eigen::VectorXd Network::costDerivative(Eigen::VectorXd outputActivations, Eigen::VectorXd y){
    return outputActivations - y;
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

Eigen::VectorXd Network::feedForward(Eigen::VectorXd prevLayerOutputs){
    for(size_t i = 0; i < biases.size(); i++){
        Eigen::VectorXd z = weights[i] * prevLayerOutputs + biases[i];
        prevLayerOutputs = sigmoid(z);
    }
    return prevLayerOutputs;
}

std::pair<std::vector<Eigen::VectorXd>,std::vector<Eigen::MatrixXd>> Network::backpropagation(Eigen::VectorXd x, Eigen::VectorXd y){ 
    std::vector<Eigen::MatrixXd> nablaWeight;
    std::vector<Eigen::VectorXd> nablaBias;
    for (const auto& b : biases) {
        nablaBias.push_back(Eigen::VectorXd::Zero(b.size()));
    }
    for (const auto& w : weights) {
        nablaWeight.push_back(Eigen::MatrixXd::Zero(w.rows(),w.cols()));
    }
    Eigen::VectorXd activation = x; //backprop alg step 1 input
    std::vector<Eigen::VectorXd> activations = {x};
    std::vector<Eigen::VectorXd> weightedInputLayers;
    for (size_t i = 0; i < biases.size(); i++){ //backprop alg step 2 feedforward
        Eigen::VectorXd z = weights[i] * activation + biases[i];
        weightedInputLayers.push_back(z);
        activation = sigmoid(z);
        activations.push_back(activation);
    }
    //backprop alg step 3 compute delta error
    Eigen::VectorXd deltaError = (costDerivative(activations[activations.size()-1], y).array() * sigmoidPrime(weightedInputLayers[weightedInputLayers.size()-1]).array()).matrix();
    nablaBias[nablaBias.size()-1] = deltaError;
    nablaWeight[nablaWeight.size()-1] = deltaError * activations[activations.size()-2].transpose();
    //backprop alg step 4 backprop the error
    for(size_t i = activations.size()-2; i >= 0; i--){
        Eigen::VectorXd z = weightedInputLayers[i];
        Eigen::VectorXd sigPrime = sigmoidPrime(z);
        deltaError = (weights[i+1].transpose() * deltaError) * sigPrime;
        //backprop step 5 output
        nablaBias[i] = deltaError;
        nablaWeight[i] = deltaError * activations[i-1].transpose(); 
    }
    return {nablaBias,nablaWeight};
}


void Network::updateMiniBatch(const std::vector<std::pair<Eigen::VectorXd,Eigen::VectorXd>> batch, double lR){
    std::vector<Eigen::VectorXd> nablaBias;
    std::vector<Eigen::MatrixXd> nablaWeight;
    for (const auto& b : biases) {
        nablaBias.push_back(Eigen::VectorXd::Zero(b.size()));
    }
    for (const auto& w : weights) {
        nablaWeight.push_back(Eigen::MatrixXd::Zero(w.rows(),w.cols()));
    }
    for(size_t i = 0; i < batch.size(); i++){
        std::pair<std::vector<Eigen::VectorXd>,std::vector<Eigen::MatrixXd>> deltaNabla = backpropagation(batch[i].first, batch[i].second);
        for(size_t j = 0; j < nablaBias.size(); j++){
            nablaBias[j] += deltaNabla.first[j];
            nablaWeight[j] += deltaNabla.second[j];
        }
    }
    for (size_t i = 0; i < weights.size(); i++){
        weights[i] = weights[i] - (lR/batch.size()) * nablaWeight[i];
    }
    for (size_t i = 0; i < biases.size(); i++){
        biases[i] = biases[i] - (lR/batch.size())*nablaBias[i];
    }
}

void Network::stochasticGradientDescent(std::vector<std::pair<Eigen::VectorXd,Eigen::VectorXd>> trainingData, size_t epochs, size_t miniBatchSize,double learningRate, std::vector<std::pair<Eigen::VectorXd,Eigen::VectorXd>> testData){
    size_t trainingDataSize = trainingData.size();
    std::random_device rd;
    std::mt19937 g(rd());
    for(size_t i = 0; i < epochs; i++){
        std::shuffle(trainingData.begin(), trainingData.end(),g);
        std::vector<std::vector<std::pair<Eigen::VectorXd,Eigen::VectorXd>>> miniBatches;
        for(size_t k = 0; k < trainingData.size(); k+=miniBatchSize){
            std::vector<std::pair<Eigen::VectorXd,Eigen::VectorXd>> batch(trainingData.begin()+k, trainingData.begin()+k+miniBatchSize);
            miniBatches.push_back(batch);
        }
        for(const auto& miniBatch : miniBatches){
            updateMiniBatch(miniBatch,learningRate);
        }
        if(testData.size() != 0){
            std::cout<< "Epoch: " << i+1 <<": " << evaluate(trainingData) << std::endl;
        }
        else{
            std::cout <<"Epoch 0 complete" << std::endl;
        }
    }
}

double Network::evaluate(std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> trainingData){
    std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> results;
    double correctCount = 0;
    for(size_t i = 0; i < trainingData.size(); i++){
        results.push_back({feedForward(trainingData[i].first),trainingData[i].second});
        int maxIndexResult = 0;
        int maxIndexExpected = 0;
        for (size_t j = 1; j < results[i].first.size(); j++){
            if (results[i].first[j] > results[i].first[maxIndexResult]){
                maxIndexResult = j;
            }
            if(results[i].second[j] > results[i].second[maxIndexExpected]){
                maxIndexExpected = j;
            }
        }
        if(maxIndexResult == maxIndexExpected){correctCount++;}
    }
    return correctCount / trainingData.size();
}