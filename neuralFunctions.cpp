#include "neuralFunctions.h"

double reLuPrime(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

Eigen::MatrixXd softMax(const Eigen::MatrixXd& Z) {
    Eigen::MatrixXd result(Z.rows(), Z.cols());
    for (int col = 0; col < Z.cols(); col++) {
        Eigen::VectorXd z = Z.col(col);
        double maxCoeff = z.maxCoeff();
        z = z.array() - maxCoeff;
        Eigen::VectorXd expZ = z.array().exp();
        double sumExp = expZ.sum();
        result.col(col) = expZ / sumExp;
    }
    return result;
}

Network::Network(std::vector<size_t>& sizes,
                 imagesInputAndValue& trainingData,
                 size_t mBS, size_t e, 
                 double r,
                 double lR, 
                 double drop,
                 const imagesInputAndValue& testingData) : 
                 testingData(testingData), 
                 trainingData(trainingData){

    std::random_device rd;
    std::mt19937 gen(rd());
    numLayers = sizes.size();
    inputSize = trainingData.size();
    miniBatchSize = mBS;
    epochs = e;
    learningRate = lR;
    reg = r;
    dropout = drop;
    for(size_t i = 0; i < sizes.size()-1; i++){
        std::normal_distribution<double> he(0, sqrt(2.0 / sizes[i]));
        weights.push_back(Eigen::MatrixXd(sizes[i+1], sizes[i]).unaryExpr([&](double){ return he(gen); }));
        biases.push_back(Eigen::VectorXd::Zero(sizes[i+1]));
    }
}

void Network::backPropagation(const std::vector<Eigen::MatrixXd>& batchActivations,
                              const std::vector<Eigen::MatrixXd>& zs,
                              const Eigen::MatrixXd& oneHots, 
                              size_t thisBatchSize){

    std::vector<std::pair<Eigen::MatrixXd,Eigen::MatrixXd>> gradients;
    Eigen::MatrixXd delta = batchActivations.back() - oneHots;
    Eigen::MatrixXd weightDeriv = (delta * batchActivations[batchActivations.size()-2].transpose()) / thisBatchSize;
    Eigen::MatrixXd biasDeriv = delta.rowwise().mean();
    weights.back() -= learningRate*weightDeriv;
    biases.back() -= learningRate*biasDeriv;
    //numLayers includes input and output layers, since we updated the output layer weights/biases 
    //start at numLayers-3 because numLayers-2 is the output layer's weights (amt of weights is -1 numLayers)
    for(int i = numLayers-3; i >= 0; i--){
        delta = (weights[i+1].transpose() * delta).cwiseProduct(zs[i].unaryExpr(&reLuPrime));
        weightDeriv = (delta * batchActivations[i].transpose()) / thisBatchSize;
        biasDeriv = delta.rowwise().mean();
        weights[i] = (1.0-(learningRate*reg)/inputSize) * weights[i] - learningRate* weightDeriv;
        biases[i] -= learningRate*biasDeriv;
    }
}

std::vector<Eigen::MatrixXd> Network::feedForwardOneBatch(const Eigen::MatrixXd& batch, std::vector<Eigen::MatrixXd>& zs){
    //vector will be size numLayers, each item a matrix of activation layers for a specific layer for all the images (columns) in the batch
    std::vector<Eigen::MatrixXd> allBatchActivations = {batch};
    Eigen::MatrixXd batchActivationMatrix = batch; //assign first input for each image vector in matrix
    for(size_t l = 0; l < numLayers - 2; l++){
        batchActivationMatrix = (weights[l] * batchActivationMatrix).colwise() + biases[l];
        zs.push_back(batchActivationMatrix);
        batchActivationMatrix = batchActivationMatrix.array().max(0); //reLu
        //dropout
        Eigen::MatrixXd mask = Eigen::MatrixXd::Zero(batchActivationMatrix.rows(),batchActivationMatrix.cols());
        for(size_t r = 0; r < mask.rows(); r++){
            for (size_t c = 0; c < mask.cols(); c++){
                mask(r,c) = (((double)rand())/ RAND_MAX) > dropout ? 1.0 / (1.0-dropout) : 0.0;
            }
        }
        allBatchActivations.push_back(batchActivationMatrix);
    }
    //softmax last matrix of activations (representing last layer of activations across all images in batch)
    batchActivationMatrix = (weights.back() * batchActivationMatrix).colwise() + biases.back();
    zs.push_back(batchActivationMatrix);
    batchActivationMatrix = softMax(batchActivationMatrix);
    allBatchActivations.push_back(batchActivationMatrix);
    return allBatchActivations;
}

void Network::testNetwork(){
    size_t correctCount = 0;
    std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd>> results;
    for(size_t i = 0; i < testingData.size(); i++){
        std::vector<Eigen::MatrixXd> zTest;
        //could do batches of testing, rn should be a bunch of column vectors
        results.push_back({feedForwardOneBatch(testingData[i].first,zTest).back(), testingData[i].second});
        int maxIndexResult = 0;
        int maxIndexExpected = 0;
        for (size_t j = 1; j < results[i].first.rows(); j++){
            if (results[i].first(j,0) > results[i].first(maxIndexResult,0)){
                maxIndexResult = j;
            }
            if(results[i].second(j) > results[i].second(maxIndexExpected)){
                maxIndexExpected = j;
            }
        }
        if(maxIndexResult == maxIndexExpected){correctCount++;}
    }
    std::cout << correctCount << " / " << testingData.size() << std::endl;
}

void Network::sgdTrain(){
    auto rng = std::default_random_engine {};
    for(size_t i = 0; i < epochs; i++){
        std::shuffle(trainingData.begin(),trainingData.end(), rng);
        size_t j = 0;
        while(j < inputSize){
            size_t end = j + miniBatchSize;
            if(end >= inputSize){
                end = inputSize;
            }
            size_t thisBatchSize = end-j;
            Eigen::MatrixXd batchInputs(784, thisBatchSize);
            Eigen::MatrixXd oneHots(10,thisBatchSize);
            for(size_t k = j; k < end; k++){
                size_t batchIndex = k - j;
                batchInputs.col(batchIndex) = trainingData[k].first;
                oneHots.col(batchIndex) = trainingData[k].second;
            }
            std::vector<Eigen::MatrixXd> zs;
            std::vector<Eigen::MatrixXd> batchActivations = feedForwardOneBatch(batchInputs,zs);
            backPropagation(batchActivations, zs, oneHots, thisBatchSize);
            j+= miniBatchSize;
        }
        time_t timestamp;
        time(&timestamp);
        std::cout << ctime(&timestamp) << ": ";
        std::cout << "Epoch " << i << ": ";
        testNetwork();
        std::cout << std::endl;
    }
}




