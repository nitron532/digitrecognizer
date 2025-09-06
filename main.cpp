#include <fstream>
#include <stdexcept>
#include <cstdlib>
#include "neuralFunctions.h"

//chatgpt generated loader
typedef std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> stdVectorPairEigVector;
// Helper: read big-endian 32-bit integer
int readInt(std::ifstream& f) {
    unsigned char bytes[4];
    f.read((char*)bytes, 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

// Load MNIST images
std::vector<Eigen::VectorXd> load_mnist_images(const std::string& filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open file: " + filename);

    int magic = readInt(f);
    int num_images = readInt(f);
    int rows = readInt(f);
    int cols = readInt(f);

    if (magic != 2051) throw std::runtime_error("Invalid MNIST image file!");

    std::vector<Eigen::VectorXd> images(num_images, Eigen::VectorXd(rows * cols));
    for (int i = 0; i < num_images; i++) {
        for (int j = 0; j < rows * cols; j++) {
            unsigned char pixel = 0;
            f.read((char*)&pixel, 1);
            images[i](j) = pixel / 255.0; // normalize [0,1]
        }
    }

    return images;
}

// Load MNIST labels
std::vector<int> load_mnist_labels(const std::string& filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open file: " + filename);

    int magic = readInt(f);
    int num_labels = readInt(f);

    if (magic != 2049) throw std::runtime_error("Invalid MNIST label file!");

    std::vector<int> labels(num_labels);
    for (int i = 0; i < num_labels; i++) {
        unsigned char label = 0;
        f.read((char*)&label, 1);
        labels[i] = (int)label;
    }

    return labels;
}

// Optional: one-hot encode labels
std::vector<Eigen::VectorXd> one_hot_encode(const std::vector<int>& labels, int num_classes = 10) {
    std::vector<Eigen::VectorXd> encoded(labels.size(), Eigen::VectorXd::Zero(num_classes));
    for (size_t i = 0; i < labels.size(); i++) {
        encoded[i](labels[i]) = 1.0;
    }
    return encoded;
}

int main(){
    std::vector<Eigen::VectorXd> trainingImages = load_mnist_images("data/trainingImages/train-images.idx3-ubyte");
    std::vector<Eigen::VectorXd> trainingLabels = one_hot_encode(load_mnist_labels("data/train-labels.idx1-ubyte"));
    std::vector<Eigen::VectorXd> testingImages =  load_mnist_images("data/testingImages/t10k-images.idx3-ubyte");
    std::vector<Eigen::VectorXd> testingLabels = one_hot_encode(load_mnist_labels("data/t10k-labels.idx1-ubyte"));
    stdVectorPairEigVector trainingData;
    stdVectorPairEigVector testingData;
    for(size_t i = 0; i < trainingImages.size(); i++){
        trainingData.push_back({trainingImages[i],trainingLabels[i]});
        // !assuming trainingData is larger than testing data!
        if(i < testingImages.size()){
            testingData.push_back({testingImages[i],testingLabels[i]});
        }
    }
    
    std::cout << "miniBatchSize?" << std::endl; 
    int miniBatchSize = 0;
    std::cin >> miniBatchSize;
    if (miniBatchSize <= 0){
        std::cerr << "Mini batch size must be positive" << std::endl;
    }
    std::cout << "Epochs?" << std::endl; 
    int epochs = 0;
    std::cin >> epochs;
    if (epochs <= 0){
        std::cerr << "Epochs must be positive" << std::endl;
    }
    std::cout << "How many hidden layers?" << std::endl;
    int layerCount = 0;
    std::cin >> layerCount;
    std::vector<size_t> layerVector = {784};
    if(layerCount < 0){
        std::cerr << "Layer count must be nonnegative.";
    }
    for( size_t i = 0; i < layerCount; i++){
        std::cout << "For layer " << i+1 << ", how many neurons?" << std::endl;
        int neurons = 0;
        std::cin >> neurons;
        if(neurons <= 0){
            std::cerr << "Must have atleast one neuron in each layer." << std::endl;
        }
        layerVector.push_back((size_t)neurons);
    }
    layerVector.push_back(10);
    std::cout << "Learning rate?" << std::endl;
    double learningRate = 0;
    std::cin >> learningRate;
    if (learningRate <= 0){
        std::cerr << "Learning rate must be positive" << std::endl;
    }
    std::cout << "Regularization?" << std::endl;
    double reg = 0;
    std::cin >> reg;
    if (reg <= 0){
        std::cerr << "Regularization must be positive" << std::endl;
    }
    std::cout << "Dropout?" << std::endl;
    double drop = 0;
    std::cin >> drop;
    if (drop < 0 || drop >= 1){
        std::cerr << "Dropout probability must be between [0,1)" << std::endl;
    }
    std::cout<< "Constructed neural network! " << std::endl;
    Network brain = Network(layerVector,trainingData, miniBatchSize, epochs, reg, drop, learningRate, testingData);
    time_t timestamp;
    time(&timestamp);
    std::cout << "Beginning stochastic gradient descent at " << ctime(&timestamp) << std::endl;
    srand(time(0));
    brain.sgdTrain();
    return 0;
}