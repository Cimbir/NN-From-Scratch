#include "matrix.hpp"

#include <iostream>
#include <vector>
#include <cmath>

#define entry pair<Matrix, Matrix>

using namespace std;

class LinearLayer {
public:
Matrix weights;
Matrix output;
    
    LinearLayer(int inputSize, int outputSize){
        weights = Matrix(outputSize, inputSize);
        
        output = Matrix(outputSize);
    }

    Matrix* forward(Matrix& input){
        weights.prod(input, output);
        return &output;
    }
};

class ActivationLayer{
public:    
Matrix output;
Matrix output_d;

    ActivationLayer(int size){
        output = Matrix(size);
        output_d = Matrix(size);
    }

    Matrix* forward(Matrix& input){
        // ReLU
        // for(int i = 0; i < input.size(0); i++){
        //     output(i) = input(i) > 0 ? input(i) : 0;
        // }
        // Sigmoid
        for(int i = 0; i < input.size(0); i++){
            output(i) = 1 / (1 + exp(-input(i)));
        }
        return &output;
    }

    Matrix* forward_d(Matrix& input){
        // ReLU derivative
        // for(int i = 0; i < input.size(0); i++){
        //     output_d(i) = input(i) > 0 ? 1 : 0;
        // }
        // Sigmoid derivative
        for(int i = 0; i < input.size(0); i++){
            output_d(i) = output(i) * (1 - output(i));
        }
        return &output_d;
    }
};

class Error{
public:
double output;
Matrix output_d;

    Error() {
        output = 0;
    }

    Error(int size){
        output_d = Matrix(size);
    }

    double forward(Matrix& input, Matrix& target){
        // ! WRONG
        return input.dot(target) * 1/2;
    }

    Matrix* forward_d(Matrix& input, Matrix& target){
        for(int i = 0; i < input.size(0); i++){
            output_d(i) = target(i) - input(i);
        }
        return &output_d;
    }
};

class NeuralNetwork{
private:
Matrix inp;
vector<Matrix> y_deltas;
vector<Matrix> weight_updates;

public:
vector<LinearLayer> linears;
vector<ActivationLayer> activations;
Error error;

    NeuralNetwork(vector<int> sizes){
        sizes[0] += 1; // * Bias
        for(int i = 0; i < (int)sizes.size() - 1; i++){
            linears.push_back(LinearLayer(sizes[i], sizes[i + 1]));
            activations.push_back(ActivationLayer(sizes[i + 1]));
            
            y_deltas.push_back(Matrix(sizes[i + 1]));
            weight_updates.push_back(Matrix(sizes[i + 1], sizes[i]));
        }
        error = Error(sizes[sizes.size() - 1]);

        inp = Matrix(sizes[0]);
        inp(sizes[0]-1) = 1;
    }

    Matrix* forward(Matrix& input){
        for(int i = 0; i < input.size(0); i++){
            inp(i) = input(i);
        }
        Matrix* out = &inp;
        for(int i = 0; i < (int)linears.size(); i++){
            out = activations[i].forward(*linears[i].forward(*out));
        }
        return out;
    }

    void backward(Matrix& input, Matrix& target){
        Matrix* out = forward(input);
        Matrix* err = error.forward_d(*out, target);

        // * Y output deltas
        for(int i = linears.size() - 1; i >= 0; i--){
            Matrix* fd = activations[i].forward_d(linears[i].output);
            if(i == (int)linears.size() - 1){
                (*err).mult(*fd, y_deltas[i]);
            } else {
                linears[i+1].weights.prod(y_deltas[i+1], y_deltas[i], 2);
                y_deltas[i].mult(*fd, y_deltas[i]);
            }
        }

        // * Weight Update
        for(int i = 0; i < (int)linears.size(); i++){
            if(i == 0){
                y_deltas[i].prod(inp, weight_updates[i], 1);
            }else{
                y_deltas[i].prod(activations[i-1].output, weight_updates[i], 1);
            }
            linears[i].weights.add(weight_updates[i], linears[i].weights, 0.1);
        }
    }

    void train(vector<entry>& data, int epochs){
        for(int i = 0; i < epochs; i++){
            for(int j = 0; j < (int)data.size(); j++){
                backward(data[j].first, data[j].second);
            }
        }
    }
};


vector<entry> circleData(double range, double x, double y, double r, int amount){
    vector<entry> data;
    srand(time(0));
    for(int i = 0; i < amount; i++){
        double cx = ((double)(rand() % 100) / 100) * range;
        double cy = ((double)(rand() % 100) / 100) * range;
        double target = (cx - x) * (cx - x) + (cy - y) * (cy - y) < r * r ? 1 : 0;

        Matrix dataPoint(2);
        dataPoint(0) = cx;
        dataPoint(1) = cy;

        Matrix targetPoint(2);
        targetPoint(0) = target;
        targetPoint(1) = 1 - target;

        data.push_back(entry(dataPoint, targetPoint));
    }
    return data;
}

int main(){
    srand(time(0));
    NeuralNetwork nn({2, 20, 2});

    vector<entry> trainingData = circleData(10, 5, 5, 3, 1000);
    vector<entry> testData = circleData(10, 5, 5, 3, 10);

    while(true){
        nn.train(trainingData, 100);

        for(int i = 0; i < (int)testData.size(); i++){
            Matrix* out = nn.forward(testData[i].first);
            cout << "Input: " << testData[i].first.to_string() << " Target: " << testData[i].second.to_string() << " Output: " << out->to_string() << endl;
        }
    }

    return 0;
}