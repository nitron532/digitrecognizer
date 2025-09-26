#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <algorithm>
#include <random>
#include <ctime>
#include <fstream>
#include "ThreadPool.h"
/*
This class implements a neural network which uses mini-batch stochastic gradient descent.
Loss - Categorical Cross Entropy
Activations - Softmax for output, ReLu for hidden
Initialization - He
L2 Regularization
Dropout
*/

/*
imagesInputAndValue
A list of all images in the training/testing data.
Stores a pair, where
first -> the actual pixel values, a column vector of 784 x 1
second -> the actual digit the image represents, a onehot column vector of 10 x 1
*/
typedef std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> imagesInputAndValue;

class Network{
    private:
        size_t inputSize = 1; //how many images to train with
        size_t numLayers = 2; //total amount of layers
        size_t miniBatchSize = 1; //chosen mini-batch size
        size_t epochs = 1; //how many epochs to train for
        double learningRate = 1;
        double reg = 1; //lambda aka regularization parameter
        double dropout = 0;
        ThreadPool threadPool;
        std::vector<Eigen::MatrixXd> weights;
        std::vector<Eigen::VectorXd> biases;
        imagesInputAndValue& trainingData; //non-const since it's shuffled for SGD
        const imagesInputAndValue& testingData;
        /*
        feedForwardOneBatch()
        feeds forward a matrix of 784 x miniBatchSize, where each column is an image 
        zs is passed in by reference to record weighted inputs pre-activation, where each column of a matrix is an image's weighted inputs at that layer
        if the last batch isnt of miniBatchSize (for example if inputSize = 67 and miniBatchSize = 32), the amount of columns is the remainder
        returns a vector of matrices, where each matrix corresponds to a layer, and each column of that matrix is a specific image's activation from that layer
        */
        std::vector<Eigen::MatrixXd> feedForwardOneBatch(const Eigen::MatrixXd& batch, std::vector<Eigen::MatrixXd>& zs);

        /*
        backPropagation()
        calculates deltas for batchActivations (one batch) and updates weights and biases
        batchActivations is the return value from feedForwardOneBatch()
        zs is the modified parameter from feedForwardOneBatch()
        oneHots is a 10 x miniBatchSize (or remaining images size) matrix, where each column of a matrix is an image's respective digit in a onehot vector
        thisBatchSize is the size of the batch (usually miniBatchSize, though could be remainder)
        */
        void backPropagation(const std::vector<Eigen::MatrixXd>& batchActivations,
                             const std::vector<Eigen::MatrixXd>& zs, 
                             const Eigen::MatrixXd& oneHots, 
                             size_t thisBatchSize);
        

        /*
        crossEntropyLoss()
        Calculates loss per backprop. Used to return value to monitoring modules.
        */
        double crossEntropyLoss(const Eigen::MatrixXd& softMaxActivations, const Eigen::MatrixXd& expectedOutputs);
        /*
        threadTrain()
        Feedforward and backprop allotted amount of batches for one thread using a thread.
        startIndex is the index corresponding to the first image in trainingData that the thread will work on.
        endIndex is the index corresponding to the image after the last image in trainingData that the thread will work on.
        mtx is the mutex lock used to lock backpropagation and the logging of cost values.
        Essentially, the thread works on trainingData[startIndex] to trainingData[endIndex-1].
        */
        void threadTrain(size_t startIndex, size_t endIndex, size_t& taskCounter);
    public:
        Network(std::vector<size_t>& sizes, 
                imagesInputAndValue& trainingData, 
                size_t mBS, 
                size_t e, 
                double r,
                double lR, 
                double drop,
                const imagesInputAndValue& testingData);
        /*
        sgdTrain()
        runs feed forwarding then backpropagation functions for each minibatch created, epochs amount of times
        runs testNetwork at the end of each epoch to evaluate accuracy
        */
        void sgdTrain();
        /*
        testNetwork()
        feeds forward test images through the trained network, and compares the final activation with the actual onehot vector value (no backprop done since we're not training on test data)
        returns amount of correctly identified images
        */
        size_t testNetwork();
};