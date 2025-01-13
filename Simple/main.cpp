#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <cmath>
#include <thread>
#include <chrono>

using namespace std;

// ================== Data Structure ==================

struct Vec {
    vector<double> data;
    Vec() {}
    Vec(vector<double> data) {
        this->data = vector<double>(data);
    }
    Vec(int size) {
        this->data = vector<double>(size);
    }
    Vec(initializer_list<double> list) {
        this->data = vector<double>(list);
    }
    double& operator[](size_t index) {
        return data[index];
    }
    int size() {
        return data.size();
    }
    double operator*(Vec& other) {
        double result = 0;
        for(int i = 0; i < data.size(); i++){
            result += data[i] * other[i];
        }
        return result;
    }
    string asString() {
        string res = "[";
        for(int i = 0; i < data.size(); i++){
            res += to_string(data[i]);
            if(i < data.size() - 1){
                res += ", ";
            }
        }
        res += "]";
        return res;
    }
} typedef Vec;

// ================== Activation Functions ==================

class ActivationFunction {
public:
    virtual double operator()(double x) const = 0;
    virtual double derivative(double x, double y) const = 0;
    virtual unique_ptr<ActivationFunction> clone() const = 0;
    virtual ~ActivationFunction() = default;
} typedef ActivationFunction;

class ReLU: public ActivationFunction {
public:
    double operator()(double x) const override {
        return max(0.0, x);
    }
    double derivative(double x, double y) const override {
        return x > 0 ? 1 : 0;
    }
    unique_ptr<ActivationFunction> clone() const override {
        return make_unique<ReLU>(*this);
    }
} typedef ReLU;

class Sigmoid: public ActivationFunction {
public:
    double operator()(double x) const override{
        return 1 / (1 + exp(-x));
    }
    double derivative(double x, double y) const override{
        return y * (1 - y);
    }
    unique_ptr<ActivationFunction> clone() const override {
        return make_unique<Sigmoid>(*this);
    }
} typedef Sigmoid;

// ================== Neural Network ==================

class Layer {
public:
    vector<Vec> weights;
    Layer() {}
    Layer(int input, int output) {
        this->weights = vector<Vec>(output, Vec(input));
        for(int i = 0; i < output; i++){
            for(int j = 0; j < input; j++){
                weights[i][j] = (rand() % 100) / 100.0;
            }
        }
    }

    Vec forward(Vec& input) {
        Vec res(weights.size());
        for(int i = 0; i < weights.size(); i++){
            res[i] = input * weights[i];
        }
        return res;
    }

    string asString() {
        string res = "[";
        for(int i = 0; i < weights.size(); i++){
            res += weights[i].asString();
            if(i < weights.size() - 1){
                res += ",\n";
            }
        }
        res += "]";
        return res;
    }
} typedef Layer;

class ActivationLayer{
public:
    unique_ptr<ActivationFunction> activation;

    Vec before;
    Vec after;
    ActivationLayer() {}
    ActivationLayer(int input, unique_ptr<ActivationFunction> activation) {
        this->activation = move(activation);
        this->before = Vec(input);
        this->after = Vec(input);
    }

    Vec forward(Vec& input) {
        before = input.data;
        for(int i = 0; i < input.data.size(); i++){
            after[i] = (*activation)(input[i]);
        }
        return after;
    }
    Vec backward(Vec& actual, Vec& target) {
        Vec delta(actual.size());
        for(int i = 0; i < actual.size(); i++){
            delta[i] = activation->derivative(before[i], actual[i]) * (target[i] - actual[i]);
        }
        return delta;
    }
    Vec backward(Vec& nextDelta, Layer& weights) {
        Vec delta(before.size());
        for(int i = 0; i < before.size(); i++){
            delta[i] = 0;
            for(int j = 0; j < nextDelta.size(); j++){
                delta[i] += nextDelta[j] * weights.weights[j][i];
            }
            delta[i] *= activation->derivative(before[i], after[i]);
        }
        return delta;
    }
} typedef ActivationLayer;



struct NeuralNetwork {
    vector<Layer> layers;
    vector<ActivationLayer> activations;
    double lr;

    NeuralNetwork(vector<int> sizes, unique_ptr<ActivationFunction> activation, double lr) {
        this->layers = vector<Layer>(sizes.size() - 1);
        for(int i = 0; i < layers.size(); i++){
            layers[i] = Layer(sizes[i], sizes[i + 1]);
        }

        this->activations = vector<ActivationLayer>(sizes.size() - 1);
        for(int i = 0; i < activations.size(); i++){
            activations[i] = ActivationLayer(sizes[i + 1], activation->clone());
        }
        this->lr = lr;
    }

    Vec forward(Vec input) {
        for(int i = 0; i < layers.size(); i++){
            input = layers[i].forward(input);
            //cout << "Layer " << i << ":\n" << layers[i].asString() << endl;
            input = activations[i].forward(input);
        }
        return input;
    }

    void backward(Vec& input, Vec& target) {
        Vec output = forward(input);

        int n = layers.size();
        vector<Vec> deltas(n);
        for(int i = 0; i < n; i++){
            deltas[i] = vector<double>(layers[i].weights.size());
        }


        deltas[n-1] = activations[n-1].backward(output, target);
        for(int i = layers.size() - 2; i >= 0; i--){
            deltas[i] = activations[i].backward(deltas[i+1], layers[i + 1]);
        }

        // cout << "=====================" << endl;
        // for(int i = 0; i < deltas.size(); i++){
        //     for(int j = 0; j < deltas[i].size(); j++){
        //         cout << deltas[i][j] << " ";
        //     }
        //     cout << endl;
        // }
        // cout << "=====================" << endl;

        for(int i = 0; i < layers.size(); i++){
            for(int j = 0; j < layers[i].weights.size(); j++){
                for(int k = 0; k < layers[i].weights[j].size(); k++){
                    if(i == 0){
                        layers[i].weights[j][k] += lr * deltas[i][j] * input[k];
                    }else{
                        layers[i].weights[j][k] += lr * deltas[i][j] * activations[i-1].after[k];
                    }
                }
            }
        }
    }

    void train(vector<Vec>& dataset, vector<Vec>& resultset, int epochs) {
        for(int i = 0; i < epochs; i++){
            for(int j = 0; j < dataset.size(); j++){
                backward(dataset[j], resultset[j]);
            }
        }
    }

} typedef NeuralNetwork;

pair<vector<Vec>,vector<Vec>> generateLineDataset() {
        vector<Vec> dataset = {
        {1, -1, 3},
        {1, -1, 1},
        {1, 1, -2},
        {1, 2, -3},
        {1, 2, 6},
        {1, 6, 2},
        {1, 6, -1}
    };

    vector<Vec> resultset = {
        {1},
        {1},
        {1},
        {1},
        {0},
        {0},
        {0}
    };

    return {dataset, resultset};
}

pair<vector<Vec>,vector<Vec>> generateCircleDataset() {
    vector<Vec> dataset;
    vector<Vec> resultset;
    for(int i = 0; i < 1000; i++){
        double x = (rand() % 1000) / 100.0;
        double y = (rand() % 1000) / 100.0;
        dataset.push_back({1, x, y});
        resultset.push_back({(x-5) * (x-5) + (y-5) * (y-5) < 9 ? 1.0 : 0.0});
    }
    return {dataset, resultset};
}

int main(){
    srand(time(0));
    auto [dataset, resultset] = generateCircleDataset();

    unique_ptr<ActivationFunction> sigmoid = make_unique<Sigmoid>();
    NeuralNetwork nn({3, 8, 8, 1}, move(sigmoid), 0.1);

    while(true){
        cout << "training..." << endl;
        nn.train(dataset, resultset, 100);
        for(int i = 0; i < 10; i++){
            double x = dataset[i][1];
            double y = dataset[i][2];
            double dist = (x-5) * (x-5) + (y-5) * (y-5);
            cout << "( " << dataset[i][1] << ", " << dataset[i][2] << " )  " << dist << " -> " << nn.forward(dataset[i])[0] << endl;
        }
        cout << "=====================" << endl;
        this_thread::sleep_for(chrono::seconds(1));
    }

    return 0;
}