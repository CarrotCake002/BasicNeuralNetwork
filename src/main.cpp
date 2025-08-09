#include "../include/NeuralNetwork.hpp"

std::vector<std::vector<double>> inputs = {
    {0.0, 0.0},
    {0.0, 1.0},
    {1.0, 0.0},
    {1.0, 1.0}
};

std::vector<std::vector<double>> targets = {
    {0.0},
    {0.0},
    {0.0},
    {1.0}
};

int main(int argc, char *argv[]) {
    NeuralNetwork nn({2, 2, 1});
    std::string pathToModels = "../models/";
    std::string fileExtension = ".bin";

    if (argc > 1) {
        std::string mode = argv[1];
        
        if (mode == "train") {
            int maxEpochs = 1000000;
            double learningRate = 0.1;
            std::vector<double> outputs;

            for (int epoch = 0; epoch < maxEpochs; epoch++) {
                double totalLoss = 0.0;
                for (size_t i = 0; i < inputs.size(); i++) {

                    std::vector<double> prediction = nn.forward(inputs[i]);
                    totalLoss += nn.meanSquaredError(prediction, targets[i]);
                    nn.backPropagate(inputs[i], targets[i], learningRate);
                }
                if (epoch % 100000 == 0)
                    std::cout << "Epoch: " << epoch << " - Loss: " << totalLoss / inputs.size() << "\n";
            }
            if (argc > 2) {
                nn.saveModel(pathToModels + argv[2] + fileExtension);
                std::cout << "Model saved to " << argv[2] << fileExtension << std::endl;
            } else {
                std::cerr << "Please provide a filename to save model.\n";
                return -1;
            }
        }
        else if (mode == "test") {
            if (argc > 2) {
                nn.loadModel(pathToModels + argv[2] + fileExtension);
                std::cout << "Model loaded from " << argv[2] << fileExtension << std::endl;

                for (const auto& input : inputs) {
                    auto output = nn.forward(input);
                    std::cout << "Input: (" << input[0] << ", " << input[1]
                              << ") Output: " << output[0] << std::endl;
                }
            }
            else
                std::cerr << "Please provide a filename to load model.\n";
                return -1;
        } else
            std::cerr << "Unknown mode: " << mode << ". Use 'train' or 'test'.\n";
            return -1;
    } else {
        std::cout << "Usage:\n";
        std::cout << argv[0] << " train [save_filename]\n";
        std::cout << argv[0] << " test  [load_filename]\n";
    }
    return 0;
}