#include "neuralFunctions.h"
#include <omp.h>

double reLuPrime(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

// void logger(std::string line, std::string fileName){
//     std::ofstream outfile;
//     outfile.open(fileName + ".txt", std::ios_base::app);
//     outfile << line << '\n';
// }

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
                 trainingData(trainingData),inputSize(trainingData.size()), testSize(testingData.size()){

    std::random_device rd;
    std::mt19937 gen(rd());
    numLayers = sizes.size();
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

double Network::crossEntropyLoss(const Eigen::MatrixXd& softMaxActivations, const Eigen::MatrixXd& expectedOutputs){
    Eigen::MatrixXd crossEntropyLossMatrix = (softMaxActivations.unaryExpr(&log)).cwiseProduct(expectedOutputs);
    return ((-1.0/crossEntropyLossMatrix.cols()) * crossEntropyLossMatrix.sum());
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
        // Eigen::MatrixXd mask = Eigen::MatrixXd::Zero(batchActivationMatrix.rows(),batchActivationMatrix.cols());
        // for(size_t r = 0; r < mask.rows(); r++){
        //     for (size_t c = 0; c < mask.cols(); c++){
        //         mask(r,c) = (((double)rand())/ RAND_MAX) > dropout ? 1.0 / (1.0-dropout) : 0.0;
        //     }
        // }
        allBatchActivations.push_back(batchActivationMatrix);
    }
    //softmax last matrix of activations (representing last layer of activations across all images in batch)
    batchActivationMatrix = (weights.back() * batchActivationMatrix).colwise() + biases.back();
    zs.push_back(batchActivationMatrix);
    batchActivationMatrix = softMax(batchActivationMatrix);
    allBatchActivations.push_back(batchActivationMatrix);
    return allBatchActivations;
} 


size_t Network::testNetwork(){
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
    return correctCount;
}


void Network::threadTrain(size_t start, size_t end, std::atomic<size_t>& taskCounter){
    while(start < end){
        size_t endOfBatch = start + miniBatchSize;
        if(endOfBatch >= end){
            endOfBatch = end;
        }
        size_t thisBatchSize = endOfBatch - start;
        Eigen::MatrixXd batchInputs(784, thisBatchSize);
        Eigen::MatrixXd oneHots(10, thisBatchSize);
        for(size_t k = start; k < endOfBatch; k++){
            size_t batchIndex = k - start;
            batchInputs.col(batchIndex) = trainingData[k].first;
            oneHots.col(batchIndex) = trainingData[k].second;
        }
        std::vector<Eigen::MatrixXd> zs;
        std::vector<Eigen::MatrixXd> batchActivations = feedForwardOneBatch(batchInputs, zs);
        /*
        with mutex, weight and bias updates are applied for a wall-clock previous batch, meaning by the time the last patch gets
        processed, its updates are irrelevant because the first few threads have already moved on
        
        without mutex, sudden updates can make the loss explode when network was designed with deterministic weight/bias state in mind

        maybe could process miniBatchSize in parallel, accumulate gradients in parallel, backprop singlethreaded?

        logging commented out for optimization
        */
        // backPropLock.lock();
        backPropagation(batchActivations, zs, oneHots, thisBatchSize);
        // backPropLock.unlock();
        // if((start/miniBatchSize)%2==0){
        //     logger(std::to_string(crossEntropyLoss(batchActivations[numLayers-1], oneHots)), "cost");
        // }
        start += miniBatchSize;
    }
    taskCounter.fetch_add(1, std::memory_order_release);
}

size_t Network::threadTest(size_t start, size_t end, std::atomic<size_t>& taskCounter){
    size_t correct = 0;
    std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd>> results;
    size_t resultIndex = 0;
    while(start < end){
        std::vector<Eigen::MatrixXd> zTest;
        results.push_back({feedForwardOneBatch(testingData[start].first,zTest).back(), testingData[start].second});
        int maxIndexResult = 0;
        int maxIndexExpected = 0;
        //argmax?
        for (size_t j = 1; j < results[resultIndex].first.rows(); j++){
            if (results[resultIndex].first(j,0) > results[resultIndex].first(maxIndexResult,0)){
                maxIndexResult = j;
            }
            if(results[resultIndex].second(j) > results[resultIndex].second(maxIndexExpected)){
                maxIndexExpected = j;
            }
        }
        if(maxIndexResult == maxIndexExpected){correct++;}
        start++;
        resultIndex++;
    }
    return correct;
}


void Network::sgdTrain(){
    omp_set_num_threads(1); //disable openmp parallel matrix ops (for now)
    auto rng = std::default_random_engine {};
    size_t cores = std::thread::hardware_concurrency();
    size_t inputsPerThreadTrain = static_cast<size_t>(ceil(static_cast<double>(inputSize) / cores));
    std::mutex ensureSequential;
    std::condition_variable cv;
    for(size_t i = 0; i < epochs; i++){

        std::atomic<size_t> trainCounter = 0;
        std::atomic<size_t> testCounter = 0;

        std::shuffle(trainingData.begin(),trainingData.end(), rng);
        size_t beginIndex = 0;
        size_t endIndex = inputsPerThreadTrain;
        for(size_t j = 0; j < cores; j++){
            threadPool.enqueueTask([this, beginIndex, endIndex, &trainCounter,&ensureSequential, &cv, cores]() mutable {
                this->threadTrain(beginIndex, endIndex, trainCounter);
                if (trainCounter.fetch_add(1, std::memory_order_release) + 1 == cores) {
                    std::lock_guard<std::mutex> lk(ensureSequential);
                    cv.notify_one();
                }
            });
            beginIndex = endIndex;
            endIndex += inputsPerThreadTrain;
            if(endIndex > inputSize){
                endIndex = inputSize;
            }
        }
        {
            std::unique_lock<std::mutex> lk(ensureSequential);
            cv.wait(lk, [&] { return trainCounter.load() == cores; });
        }

        // beginIndex = 0;
        // size_t inputsPerThreadTest = static_cast<size_t>(ceil(static_cast<double>(testSize) / cores));
        // endIndex = inputsPerThreadTest;

        // std::atomic<size_t> overallCorrect = 0;
        // for(size_t j = 0; j < cores; j++){
        //     threadPool.enqueueTask([this, beginIndex, endIndex, &testCounter,&overallCorrect, &cv, &ensureSequential, cores]() mutable{
        //         size_t correct = this->threadTest(beginIndex,endIndex, testCounter);
        //         overallCorrect.fetch_add(correct, std::memory_order_relaxed);
        //         if (testCounter.fetch_add(1, std::memory_order_release) + 1 == cores) {
        //             std::lock_guard<std::mutex> lk(ensureSequential);
        //             cv.notify_one();
        //         }
        //     });
        //     beginIndex = endIndex;
        //     endIndex+= inputsPerThreadTest;
        //     if(endIndex > testSize){
        //         endIndex = testSize;
        //     }
        // }
        // {
        //     std::unique_lock<std::mutex> lk(ensureSequential);
        //     cv.wait(lk, [&] { return testCounter.load() == cores; });
        // }
        // time_t timestamp;
        // time(&timestamp);
        // std::cout << ctime(&timestamp) << ": ";
        // std::cout << "Epoch " << i << ": " << overallCorrect << "/ " << testSize;
        std::cout << "Epoch "<< i << ": "<< testNetwork() << " / " << testSize;
        std::cout << std::endl;
    }
}




