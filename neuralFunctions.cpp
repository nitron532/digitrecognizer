#include "neuralFunctions.h"
//consider passing things by reference for efficiency ( sigmoid functions)
typedef std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> stdVectorPairEigVector;
Eigen::MatrixXd sigmoid(Eigen::MatrixXd z){ //might wanna change the auto to the actual type?
    z = z.cwiseMin(50).cwiseMax(-50);
    return 1.0/(1.0 + (-z).array().exp());
} 
Eigen::MatrixXd sigmoidPrime(Eigen::MatrixXd z){
    Eigen::MatrixXd s = sigmoid(z);
    return (s.array() * (1.0 - s.array())).matrix();
}

Eigen::VectorXd Network::costDerivative(Eigen::VectorXd outputActivations, Eigen::VectorXd y){
    return outputActivations - y;
}
void logger(std::string& line){
    std::ofstream outfile;
    outfile.open("results.txt", std::ios_base::app);
    outfile << line;
}

Network::Network(std::vector<size_t>& lS){ //since numLayers is determined by user, could just pass an array to save space (instead of a vector)
    numLayers = lS.size();
    layerSizes = lS;
    for(size_t i = 1; i < numLayers; i++){
        biases.push_back(Eigen::VectorXd::Zero(lS[i]));
    }
    for(size_t i = 0; i < numLayers-1; i++){
        // weights.push_back(Eigen::MatrixXd::Random(lS[i+1],lS[i]));
        //uniform xavier initialization of weights tensor
        // weights.push_back(Eigen::MatrixXd::Random(lS[i+1], lS[i]) * sqrt(6.0/(lS[i]+lS[i+1])));
        weights.push_back(Eigen::MatrixXd::Random(lS[i+1], lS[i]) * sqrt(1.0 / lS[i]));
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
    // Eigen::VectorXd deltaError = (costDerivative(activations[activations.size()-1], y).array() * sigmoidPrime(weightedInputLayers[weightedInputLayers.size()-1]).array()).matrix();
    // deltaError = deltaError.cwiseMin(100.0).cwiseMax(-100.0);
    Eigen::VectorXd deltaError = (activations.back() - y);
    nablaBias[nablaBias.size()-1] = deltaError;
    nablaWeight[nablaWeight.size()-1] = deltaError * activations[activations.size()-2].transpose();
    //backprop alg step 4 backprop the error
    for(int i = activations.size()-2; i >= 0; i--){

        Eigen::VectorXd z = weightedInputLayers[i];
        std::cout << z << " weighted Input Z\n";
        
        Eigen::VectorXd sigPrime = sigmoidPrime(z);
        std::cout << sigPrime << " sigprime\n";
        
        deltaError = (weights[i].transpose() * deltaError).cwiseProduct(sigPrime);
        std::cout << deltaError[0] << " delta error first element\n";

        nablaBias[i] = deltaError;
        Eigen::RowVectorXd aPrevT = activations[i].transpose();
        nablaWeight[i] = deltaError * aPrevT;
    }
    return {nablaBias,nablaWeight};
}


void Network::updateMiniBatch(const stdVectorPairEigVector batch, double lR){
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
        // int test = 0;
        // std::cin >> test;
        // std::cout << weights[i] << " before stepping";
        // std:: cin >> test;
        weights[i] = weights[i] - (lR/batch.size()) * nablaWeight[i];
        //         std::cin >> test;
        // std::cout << weights[i] << " after stepping";
        // std:: cin >> test;
    }
    for (size_t i = 0; i < biases.size(); i++){
        biases[i] = biases[i] - (lR/batch.size())*nablaBias[i];
    }
}

void Network::stochasticGradientDescent(stdVectorPairEigVector trainingData, size_t epochs, size_t miniBatchSize,double learningRate, stdVectorPairEigVector testData){
    size_t trainingDataSize = trainingData.size();
    std::random_device rd;
    std::ofstream outfile;
    std::ofstream weightsFile;
    weightsFile.open("weights.txt",std::ios_base::app);
    outfile.open("results.txt", std::ios_base::app);
    std::mt19937 g(rd());
    for(size_t i = 0; i < epochs; i++){
        // std::string log = std::to_string(weights[0](0,0)) + " weight before batch update " + std::to_string(i) + " epoch\n";
        std::shuffle(trainingData.begin(), trainingData.end(),g);
        std::vector<stdVectorPairEigVector> miniBatches;
        for(size_t k = 0; k < trainingData.size(); k+=miniBatchSize){
            stdVectorPairEigVector batch(trainingData.begin()+k, trainingData.begin()+k+miniBatchSize);
            miniBatches.push_back(batch);
        }
        // logger(log);
        weightsFile << "weight before batch update " << i << "epoch \n";
        weightsFile.flush();  
        weightsFile << weights[0].row(0);
        weightsFile << "\n";
        weightsFile.flush();  
        for(const auto& miniBatch : miniBatches){
            updateMiniBatch(miniBatch,learningRate);
        }
        // log = std::to_string(weights[0](0,0)) + " weight after batch update " + std::to_string(i) + " epoch\n";
        // logger(log);
        if(testData.size() != 0){
            std::pair<size_t,size_t> metric = evaluate(trainingData);
            // std::cout<< "Epoch: " << i+1 <<": " << metric.first << " / " << metric.second << std::endl;

            outfile << "Epoch: " << i+1 <<": " << metric.first << " / " << metric.second << std::endl;
            outfile.flush();
            // log = "Epoch: " + std::to_string(i+1)+ ": " + std::to_string(metric.first) + " / " + std::to_string(metric.second)<+ "\n";
            // logger(log);
        }
        else{
            std::cout <<"Epoch 0 complete" << std::endl;
        }
    }
}

std::pair<size_t,size_t> Network::evaluate(stdVectorPairEigVector trainingData){
    stdVectorPairEigVector results;
    size_t correctCount = 0;
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
    return {correctCount, trainingData.size()};
}
