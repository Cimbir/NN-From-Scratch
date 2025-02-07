#include "matrix.hpp"

#include <iostream>
#include <vector>

using namespace std;

class LinearLayer {
public:
Matrix weights;
Matrix output;
    
    LinearLayer(int inputSize, int outputSize){
        int wd[] = {outputSize, inputSize};
        weights = Matrix(2, wd);
        
        int od[] = {outputSize};
        output = Matrix(1, od);
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
        int od[] = {size};
        output = Matrix(1, od);
        output_d = Matrix(1, od);
    }

    Matrix* forward(Matrix& input){
        // ReLU
        for(int i = 0; i < input.size(0); i++){
            output(i) = input(i) > 0 ? input(i) : 0;
        }
        return &output;
    }

    Matrix* forward_d(Matrix& input){
        // ReLU derivative
        for(int i = 0; i < input.size(0); i++){
            output_d(i) = input(i) > 0 ? 1 : 0;
        }
        return &output_d;
    }
};

class Error{
public:
double output;
Matrix output_d;

    Error(int size){
        int od[] = {size};
        output_d = Matrix(1, od);
    }

    double forward(Matrix& input, Matrix& target){
        return input.dot(target) * 1/2;
    }

    Matrix* forward_d(Matrix& input, Matrix& target){
        for(int i = 0; i < input.size(0); i++){
            output_d(i) = input(i) - target(i);
        }
        return &output_d;
    }
};

class NeuralNetwork{
private:
Matrix inp;

public:
vector<LinearLayer> linears;
vector<ActivationLayer> activations;

    NeuralNetwork(vector<int> sizes){
        for(int i = 0; i < sizes.size() - 1; i++){
            linears.push_back(LinearLayer(sizes[i] + (i==0), sizes[i + 1]));
            activations.push_back(ActivationLayer(sizes[i + 1]));
        }
        int id[] = {sizes[0]+1};
        inp = Matrix(1, id);
        inp(sizes[0]) = 1;
    }

    Matrix* forward(Matrix& input){
        for(int i = 0; i < input.size(0); i++){
            inp(i) = input(i);
        }
        Matrix* out = &inp;
        for(int i = 0; i < linears.size(); i++){
            out = activations[i].forward(*linears[i].forward(*out));
        }
        return out;
    }

    void backward(Matrix& target){
        // ! IMPLEMENT BACKPROPAGATION
    }
};

int main(){
    NeuralNetwork nn({2, 3, 1});

    int id[] = {2};
    Matrix input(1, id);
    input(0) = 1;
    input(1) = 2;

    Matrix* output = nn.forward(input);

    cout << output->to_string() << endl;

    return 0;
}