#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <iostream>
#include <vector>
#include <random>
#include <fstream>

class NeuralNetwork {
    public:
        NeuralNetwork(std::vector<int> layers);
        ~NeuralNetwork();

        std::vector<double> forward(const std::vector<double> &input);
        double meanSquaredError(const std::vector<double> &predicted, const std::vector<double> &target);
        void backPropagate(const std::vector<double> &input, const std::vector<double> &target, double learningRate);
        void saveModel(const std::string &filename);
        void loadModel(const std::string &filename);

    private:
        double randomDouble(double min = -1.0, double max = 1.0);
        double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
        double sigmoidDerivative(double x) { return x * (1 - x); }

        std::vector<int> layers;
        std::vector<std::vector<std::vector<double>>> weights;
        std::vector<std::vector<double>> biases;
        std::vector<std::vector<double>> outputs;
};

#endif // NEURALNETWORK_HPP