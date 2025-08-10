#include "../include/NeuralNetwork.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"

NeuralNetwork::NeuralNetwork(std::vector<int> layers) : layers(layers) {
    outputs.resize(layers.size());
        for (size_t i = 0; i < layers.size(); i++) {
            outputs[i].resize(layers[i]);
        }

    for (size_t i = 0; i < layers.size() - 1; i++) {
        int neuronsInPrev = layers[i];
        int neuronsInCurrent = layers[i + 1];

        double limit = std::sqrt(6.0 / (neuronsInPrev + neuronsInCurrent));

        std::vector<std::vector<double>> layerWeights;
        std::vector<double> layerBiases;

        for (int n = 0; n < neuronsInCurrent; n++) {
            std::vector<double> neuronWeights;

            for (int w = 0; w < neuronsInPrev; w++) {
                neuronWeights.push_back(randomDouble(-limit, limit));
            }
            layerWeights.push_back(neuronWeights);
            layerBiases.push_back(0.0);
        }
        weights.push_back(layerWeights);
        biases.push_back(layerBiases);
    }
}

NeuralNetwork::~NeuralNetwork() {

}

std::vector<double> NeuralNetwork::forward(const std::vector<double> &input) {
    std::vector<double> outputs = input;

    for (size_t layer = 1; layer < layers.size(); layer++) {
        std::vector<double> nextOutputs(layers[layer], 0.0);

        for (size_t neuron = 0; neuron < layers[layer]; neuron ++) {
            double sum = 0.0;

            for (size_t prevNeuron = 0; prevNeuron < layers[layer - 1]; prevNeuron++) {
                sum += outputs[prevNeuron] * weights[layer - 1][neuron][prevNeuron];
            }
            sum += biases[layer - 1][neuron];
            nextOutputs[neuron] = sum;
        }
        if (layer == layers.size() - 1) {
            nextOutputs = softmax(nextOutputs);
        } else {
            for (auto &val : nextOutputs) {
                val = sigmoid(val);
            }
        }
        outputs = nextOutputs;
    }
    return outputs;
}

double NeuralNetwork::meanSquaredError(const std::vector<double> &predicted, const std::vector<double> &target) {
    double sum = 0.0;

    for (size_t i = 0; i < predicted.size(); i++) {
        double diff = predicted[i] - target[i];
        sum += diff * diff;
    }
    return sum / predicted.size();
}

double NeuralNetwork::categoricalCrossEntropy(const std::vector<double>& output, const std::vector<double>& target) {
    double loss = 0.0;

    for (size_t i = 0; i < output.size(); i++) {
        if (target[i] == 1) {
            loss = -std::log(output[i] + 1e-15);
            break;
        }
    }
    return loss;
}

void NeuralNetwork::backPropagate(const std::vector<double> &input, const std::vector<double> &target, double learningRate) {
    // Forward pass to get activations (raw logits at output layer)
    std::vector<std::vector<double>> activations;
    activations.push_back(input);

    for (size_t layer = 1; layer < layers.size(); layer++) {
        std::vector<double> layerActivation(layers[layer], 0.0);

        for (size_t neuron = 0; neuron < layers[layer]; neuron++) {
            double z = biases[layer - 1][neuron];
            for (size_t prevNeuron = 0; prevNeuron < layers[layer - 1]; prevNeuron++) {
                z += activations[layer - 1][prevNeuron] * weights[layer - 1][neuron][prevNeuron];
            }
            if (layer == layers.size() - 1) {
                // Output layer: raw logits, no sigmoid
                layerActivation[neuron] = z;
            } else {
                // Hidden layers: sigmoid activation
                layerActivation[neuron] = sigmoid(z);
            }
        }
        activations.push_back(layerActivation);
    }

    // Calculate deltas vector, size = number of layers
    std::vector<std::vector<double>> deltas(layers.size());

    // Output layer delta: softmax + cross-entropy derivative
    std::vector<double> outputActivations = softmax(activations.back());
    deltas.back() = std::vector<double>(layers.back());

    for (size_t neuron = 0; neuron < layers.back(); neuron++) {
        deltas.back()[neuron] = outputActivations[neuron] - target[neuron];
    }

    // Backpropagate deltas through hidden layers
    for (int layer = layers.size() - 2; layer > 0; layer--) {
        deltas[layer] = std::vector<double>(layers[layer], 0.0);
        for (int neuron = 0; neuron < layers[layer]; neuron++) {
            double error = 0.0;
            for (int nextNeuron = 0; nextNeuron < layers[layer + 1]; nextNeuron++) {
                error += deltas[layer + 1][nextNeuron] * weights[layer][nextNeuron][neuron];
            }
            double output = activations[layer][neuron]; // sigmoid output
            deltas[layer][neuron] = error * sigmoidDerivative(output);
        }
    }

    // Update weights and biases
    for (size_t layer = 0; layer < weights.size(); layer++) {
        for (int neuron = 0; neuron < layers[layer + 1]; neuron++) {
            for (int prevNeuron = 0; prevNeuron < layers[layer]; prevNeuron++) {
                weights[layer][neuron][prevNeuron] -= learningRate * deltas[layer + 1][neuron] * activations[layer][prevNeuron];
            }
            biases[layer][neuron] -= learningRate * deltas[layer + 1][neuron];
        }
    }
}

void NeuralNetwork::saveModel(const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open file for saving\n";
        return;
    }

    // Save layer structure
    size_t numLayers = layers.size();
    out.write(reinterpret_cast<char*>(&numLayers), sizeof(numLayers));
    out.write(reinterpret_cast<char*>(layers.data()), numLayers * sizeof(int));

    // Save weights
    for (auto& layerWeights : weights) {
        size_t outer = layerWeights.size();
        out.write(reinterpret_cast<char*>(&outer), sizeof(outer));
        for (auto& neuronWeights : layerWeights) {
            size_t inner = neuronWeights.size();
            out.write(reinterpret_cast<char*>(&inner), sizeof(inner));
            out.write(reinterpret_cast<char*>(neuronWeights.data()), inner * sizeof(double));
        }
    }

    // Save biases
    for (auto& layerBiases : biases) {
        size_t size = layerBiases.size();
        out.write(reinterpret_cast<char*>(&size), sizeof(size));
        out.write(reinterpret_cast<char*>(layerBiases.data()), size * sizeof(double));
    }

    out.close();
}

void NeuralNetwork::loadModel(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open file for loading\n";
        return;
    }

    // Load layer structure
    size_t numLayers;
    in.read(reinterpret_cast<char*>(&numLayers), sizeof(numLayers));
    layers.resize(numLayers);
    in.read(reinterpret_cast<char*>(layers.data()), numLayers * sizeof(int));

    // Load weights
    weights.clear();
    weights.resize(numLayers - 1);
    for (size_t i = 0; i < numLayers - 1; ++i) {
        size_t outer;
        in.read(reinterpret_cast<char*>(&outer), sizeof(outer));
        weights[i].resize(outer);
        for (size_t j = 0; j < outer; ++j) {
            size_t inner;
            in.read(reinterpret_cast<char*>(&inner), sizeof(inner));
            weights[i][j].resize(inner);
            in.read(reinterpret_cast<char*>(weights[i][j].data()), inner * sizeof(double));
        }
    }

    // Load biases
    biases.clear();
    biases.resize(numLayers - 1);
    for (size_t i = 0; i < numLayers - 1; ++i) {
        size_t size;
        in.read(reinterpret_cast<char*>(&size), sizeof(size));
        biases[i].resize(size);
        in.read(reinterpret_cast<char*>(biases[i].data()), size * sizeof(double));
    }

    in.close();
}

double NeuralNetwork::randomDouble(double min, double max) {
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(min, max);
    return dist(rng);
}

std::vector<double> NeuralNetwork::softmax(const std::vector<double> &logits) {
    std::vector<double> exps(logits.size());
    double maxLogit = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;

    for (size_t i = 0; i < logits.size(); i++) {
        exps[i] = std::exp(logits[i] - maxLogit); // for numerical stability
        sum += exps[i];
    }
    for (size_t i = 0; i < exps.size(); i++) {
        exps[i] /= sum;
    }
    return exps;
}

std::vector<double> NeuralNetwork::loadAndNormalizeImage(const char *filename) {
    int width, height, channels;

    unsigned char *data = stbi_load(filename, &width, &height, &channels, 1);
    if (!data) {
        std::cerr << "Failed to load image: " << filename << std::endl;
        return {};
    }
    if (width != 28 || height != 28) {
        std::cerr << "Image must be 28x28 pixels." << std::endl;
        free(data);
        return {};
    }

    std::vector<double> normalizedPixels(width * height);

    for (int i = 0; i < width * height; i++) {
        normalizedPixels[i] = data[i] / 255.0;
    }
    free(data);
    return normalizedPixels;
}