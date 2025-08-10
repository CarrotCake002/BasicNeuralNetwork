#include "../include/NeuralNetwork.hpp"
namespace fs = std::filesystem;

void printAsciiImage(const std::vector<double>& pixels, int width = 28, int height = 28) {
    const std::string shades = " .:-=+*#%@"; // from light to dark
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double val = pixels[y * width + x]; // assuming normalized 0..1
            int shadeIndex = static_cast<int>(val * (shades.size() - 1));
            std::cout << shades[shadeIndex];
        }
        std::cout << "\n";
    }
}

void loadImagesAndTargets(std::vector<std::vector<double>> &inputs, std::vector<std::vector<double>> &targets, NeuralNetwork &nn) {
    std::string datasetPath = "/home/carrotcake/documents/projects/personal/BasicNeuralNetwork/assets/mnist_images/";

    for (const auto& entry : fs::directory_iterator(datasetPath)) {
        if (!entry.is_regular_file()) continue;

        const std::string filename = entry.path().string();
        auto inputVec = nn.loadAndNormalizeImage(filename.c_str());
        if (inputVec.empty()) continue;  // skip failed loads

        inputs.push_back(inputVec);

        // Extract label from filename: assuming "7_123.png" format, label is first char
        char labelChar = entry.path().filename().string()[entry.path().filename().string().size() - 5];
        int label = labelChar - '0';

        std::vector<double> oneHot(10, 0.0);
        if (label >= 0 && label < 10) {
            oneHot[label] = 1.0;
        } else {
            std::cerr << "Invalid label in filename: " << filename << std::endl;
            continue;
        }

        targets.push_back(oneHot);
    }
}

int main(int argc, char *argv[]) {
    NeuralNetwork nn({784, 128, 64, 10});
    std::string pathToModels = "../models/";
    std::string fileExtension = ".bin";

    if (argc > 1) {
        std::string mode = argv[1];
        
        if (mode == "train") {
            // Load all images + targets first
            std::vector<std::vector<double>> inputs;
            std::vector<std::vector<double>> targets;
            loadImagesAndTargets(inputs, targets, nn);

            int maxEpochs = 1000;
            int displayLoss = 100;
            double learningRate = 0.001;
            std::random_device rd;
            std::mt19937 g(rd());
            int batchSize = 32;
            int saveInterval = 500;

            for (int epoch = 0; epoch < maxEpochs; epoch++) {
                double totalLoss = 0.0;

                // Shuffle dataset indices
                std::vector<size_t> indices(inputs.size());
                std::iota(indices.begin(), indices.end(), 0);
                std::shuffle(indices.begin(), indices.end(), g);

                // Loop through mini-batches
                for (size_t batchStart = 0; batchStart < inputs.size(); batchStart += batchSize) {
                    int batchEnd = std::min(batchStart + batchSize, inputs.size());

                    // Accumulate gradients over this batch
                    // (if your NN doesn't support explicit gradient accumulation, do backprop per sample but average loss)
                    for (size_t idx = batchStart; idx < batchEnd; idx++) {
                        auto prediction = nn.forward(inputs[indices[idx]]);
                        totalLoss += nn.categoricalCrossEntropy(prediction, targets[indices[idx]]);
                        nn.backPropagate(inputs[indices[idx]], targets[indices[idx]], learningRate);
                    }
                }

                if (epoch % displayLoss == 0) {
                    std::cout << "Epoch: " << epoch << " - Loss: " << totalLoss / inputs.size() << "\n";
                }
                if (epoch > 0 && epoch % saveInterval == 0) {
                    std::string filename = pathToModels + "model_epoch_" + std::to_string(epoch) + fileExtension;
                    nn.saveModel(filename);
                    std::cout << "Model checkpoint saved at epoch " << epoch << " to " << filename << std::endl;
                }
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
                if (argc > 3) {
                    auto test = nn.loadAndNormalizeImage(argv[3]);
                    printAsciiImage(test);
                    nn.loadModel(pathToModels + argv[2] + fileExtension);
                    std::cout << "Model loaded from " << argv[2] << fileExtension << std::endl;

                    auto output = nn.forward(test);

                    std::cout << "Output probabilities: ";
                    size_t predictedDigit = 0;
                    double maxProb = output[0];

                    for (size_t i = 0; i < output.size(); ++i) {
                        std::cout << output[i];
                        if (i != output.size() - 1) std::cout << ", ";

                        if (output[i] > maxProb) {
                            maxProb = output[i];
                            predictedDigit = i;
                        }
                    }
                    std::cout << std::endl;

                    std::cout << "Predicted digit: " << predictedDigit 
                            << " (Confidence: " << maxProb * 100.0 << "%)" << std::endl;
                } else {
                    std::cerr << "Please provide a filename to load the image\n";
                    return -1;
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