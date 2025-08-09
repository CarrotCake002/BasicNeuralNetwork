#include "../include/NeuralNetwork.hpp"

std::vector<std::vector<double>> inputs = {
    // Pure primaries
    {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0},
    // Secondaries & blends
    {1.0, 1.0, 0.0}, {0.0, 1.0, 1.0}, {1.0, 0.0, 1.0},
    // Grays
    {0.1, 0.1, 0.1}, {0.3, 0.3, 0.3}, {0.5, 0.5, 0.5}, {0.7, 0.7, 0.7}, {0.9, 0.9, 0.9},
    // Midtones
    {0.4, 0.6, 0.2}, {0.6, 0.4, 0.2}, {0.2, 0.6, 0.4}, {0.4, 0.2, 0.6}, {0.6, 0.2, 0.4}, {0.2, 0.4, 0.6},
    // Pastels
    {0.8, 0.6, 0.7}, {0.7, 0.8, 0.6}, {0.6, 0.7, 0.8},
    // Dark tones
    {0.1, 0.0, 0.2}, {0.2, 0.1, 0.0}, {0.0, 0.2, 0.1},
    // Bright variants
    {0.9, 0.5, 0.2}, {0.5, 0.9, 0.2}, {0.2, 0.5, 0.9},
    // Weird combos
    {0.3, 0.7, 0.9}, {0.9, 0.3, 0.7}, {0.7, 0.9, 0.3},
    {0.1, 0.9, 0.7}, {0.7, 0.1, 0.9}, {0.9, 0.7, 0.1},
    // Low saturations
    {0.2, 0.2, 0.15}, {0.15, 0.2, 0.2}, {0.2, 0.15, 0.2},
    // High saturations
    {0.95, 0.1, 0.05}, {0.05, 0.95, 0.1}, {0.1, 0.05, 0.95},
    // Random floats
    {0.43, 0.28, 0.39}, {0.11, 0.55, 0.77}, {0.88, 0.44, 0.22}, {0.33, 0.66, 0.99},
    {0.76, 0.82, 0.14}, {0.24, 0.18, 0.88}, {0.67, 0.33, 0.55},
    // Earthy tones
    {0.55, 0.47, 0.36}, {0.33, 0.27, 0.19}, {0.76, 0.69, 0.55}, {0.5, 0.4, 0.3},
    // Neon vibes
    {0.8, 1.0, 0.0}, {1.0, 0.0, 0.5}, {0.0, 1.0, 0.8},
    // Dark neons
    {0.4, 0.5, 0.0}, {0.5, 0.0, 0.3}, {0.0, 0.4, 0.5},
    // Metallic-ish
    {0.7, 0.7, 0.8}, {0.6, 0.6, 0.7}, {0.8, 0.8, 0.9},
    // Some random pastels
    {0.9, 0.7, 0.8}, {0.8, 0.9, 0.7}, {0.7, 0.8, 0.9},
    // More randoms for chaos
    {0.12, 0.77, 0.43}, {0.56, 0.23, 0.89}, {0.81, 0.44, 0.12},
    {0.45, 0.56, 0.67}, {0.29, 0.81, 0.33}, {0.61, 0.19, 0.48},
    {0.71, 0.35, 0.64}, {0.48, 0.22, 0.53}, {0.57, 0.69, 0.31},
    {0.22, 0.39, 0.45}, {0.34, 0.51, 0.27}, {0.15, 0.29, 0.64},
    // Black & white extremes
    {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0},
};


std::vector<std::vector<double>> targets = {
    // Pure primaries
    {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0},
    // Secondaries & blends
    {1.0, 1.0, 0.0}, {0.0, 1.0, 1.0}, {1.0, 0.0, 1.0},
    // Grays
    {0.1, 0.1, 0.1}, {0.3, 0.3, 0.3}, {0.5, 0.5, 0.5}, {0.7, 0.7, 0.7}, {0.9, 0.9, 0.9},
    // Midtones
    {0.4, 0.6, 0.2}, {0.6, 0.4, 0.2}, {0.2, 0.6, 0.4}, {0.4, 0.2, 0.6}, {0.6, 0.2, 0.4}, {0.2, 0.4, 0.6},
    // Pastels
    {0.8, 0.6, 0.7}, {0.7, 0.8, 0.6}, {0.6, 0.7, 0.8},
    // Dark tones
    {0.1, 0.0, 0.2}, {0.2, 0.1, 0.0}, {0.0, 0.2, 0.1},
    // Bright variants
    {0.9, 0.5, 0.2}, {0.5, 0.9, 0.2}, {0.2, 0.5, 0.9},
    // Weird combos
    {0.3, 0.7, 0.9}, {0.9, 0.3, 0.7}, {0.7, 0.9, 0.3},
    {0.1, 0.9, 0.7}, {0.7, 0.1, 0.9}, {0.9, 0.7, 0.1},
    // Low saturations
    {0.2, 0.2, 0.15}, {0.15, 0.2, 0.2}, {0.2, 0.15, 0.2},
    // High saturations
    {0.95, 0.1, 0.05}, {0.05, 0.95, 0.1}, {0.1, 0.05, 0.95},
    // Random floats
    {0.43, 0.28, 0.39}, {0.11, 0.55, 0.77}, {0.88, 0.44, 0.22}, {0.33, 0.66, 0.99},
    {0.76, 0.82, 0.14}, {0.24, 0.18, 0.88}, {0.67, 0.33, 0.55},
    // Earthy tones
    {0.55, 0.47, 0.36}, {0.33, 0.27, 0.19}, {0.76, 0.69, 0.55}, {0.5, 0.4, 0.3},
    // Neon vibes
    {0.8, 1.0, 0.0}, {1.0, 0.0, 0.5}, {0.0, 1.0, 0.8},
    // Dark neons
    {0.4, 0.5, 0.0}, {0.5, 0.0, 0.3}, {0.0, 0.4, 0.5},
    // Metallic-ish
    {0.7, 0.7, 0.8}, {0.6, 0.6, 0.7}, {0.8, 0.8, 0.9},
    // Some random pastels
    {0.9, 0.7, 0.8}, {0.8, 0.9, 0.7}, {0.7, 0.8, 0.9},
    // More randoms for chaos
    {0.12, 0.77, 0.43}, {0.56, 0.23, 0.89}, {0.81, 0.44, 0.12},
    {0.45, 0.56, 0.67}, {0.29, 0.81, 0.33}, {0.61, 0.19, 0.48},
    {0.71, 0.35, 0.64}, {0.48, 0.22, 0.53}, {0.57, 0.69, 0.31},
    {0.22, 0.39, 0.45}, {0.34, 0.51, 0.27}, {0.15, 0.29, 0.64},
    // Black & white extremes
    {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0},
};

std::vector<std::vector<double>> generateTestingData(void) {
    std::vector<std::vector<double>> tests;

    const double step = 0.1;  // smaller step = bigger dataset
    for (double r = 0.0; r <= 1.0; r += step) {
        for (double g = 0.0; g <= 1.0; g += step) {
            for (double b = 0.0; b <= 1.0; b += step) {
                tests.push_back({r, g, b});
            }
        }
    }
    return tests;
}

int main(int argc, char *argv[]) {
    NeuralNetwork nn({3, 6, 6, 3});
    std::string pathToModels = "../models/";
    std::string fileExtension = ".bin";

    if (argc > 1) {
        std::string mode = argv[1];
        
        if (mode == "train") {
            int maxEpochs = 5000000;
            int displayLoss = 50000;
            double learningRate = 0.1;
            std::vector<double> outputs;

            for (int epoch = 0; epoch < maxEpochs; epoch++) {
                double totalLoss = 0.0;
                for (size_t i = 0; i < inputs.size(); i++) {

                    std::vector<double> prediction = nn.forward(inputs[i]);
                    totalLoss += nn.meanSquaredError(prediction, targets[i]);
                    nn.backPropagate(inputs[i], targets[i], learningRate);
                }
                if (epoch % displayLoss == 0)
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
                std::vector<std::vector<double>> tests = generateTestingData();
                nn.loadModel(pathToModels + argv[2] + fileExtension);
                std::cout << "Model loaded from " << argv[2] << fileExtension << std::endl;

                for (const auto& test : tests) {
                    auto output = nn.forward(test);

                    // Print the RGB input
                    std::cout << "Input: (" 
                            << test[0] << ", " 
                            << test[1] << ", " 
                            << test[2] << ") Output: ";

                    // Print all output probabilities
                    for (size_t i = 0; i < output.size(); ++i) {
                        std::cout << output[i];
                        if (i != output.size() - 1) std::cout << ", ";
                    }
                    std::cout << std::endl;
                }
            } else {
                std::cerr << "Please provide a filename to load model.\n";
                return -1;
            }
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