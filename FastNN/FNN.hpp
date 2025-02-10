#ifndef FNN_HPP
#define FNN_HPP

#include <string>

using namespace std;

// ================== Structures ==================

// Neural Network Architecture
#define Vec double*
#define Mat Vec*
#define Net Mat*

// Data Entry
#define Data_Entry pair<Vec, Vec>

// ================== Global Variables ==================

// Activation
#define _relu 1
#define _sigmoid 0

// ================== Function Definitions ==================

// Utils
string to_string(Vec v, int n);
string to_string(Mat ma, int n, int m);
string to_string(Net n, int* sizes, int am);

// Activation Functions
double relu(double x);
double relu_d(double x, double y);
double sigmoid(double x);
double sigmoid_d(double x, double y);

// ================== Neural Network Class ==================

class FNN {
public:
// Structure
int layer_n;
int* layer_sz;
int act_type;
double lr;

// Network weights
Net weights;

// Forward data
Vec* beforeActivation;
Vec* afterActivation;
// Gradient data
Vec* delta;

    // Setup
    FNN(int layer_n, int* layer_sz, int activation, double lr);
    void init();

    // Activation Functions
    double activation(double x);
    double activation_d(double x, double y);

    // Neural Network Functions
    Vec add_bias(Vec v);
    Vec forward(Vec input);
    void backward(Vec input, Vec result, double lr);
    void train(Data_Entry* dataset, int n, int epochs, double& lr);

    // Loss
    double loss(Vec output, Vec expected);
};

#endif