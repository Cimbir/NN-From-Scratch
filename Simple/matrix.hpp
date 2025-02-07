#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <string>
#include <stdexcept>

using namespace std;

class Matrix{
private:
    int dimLen;
    int* dims;
    double* data;
    
public:
    Matrix();
    Matrix(int dimLen, int* dims);

    template<typename... Args>
    double& operator()(Args... args);

    int size(int dim);
    void prod(Matrix& m, Matrix& res);
    double dot(Matrix& m);

    string to_string();
};

template<typename... Args>
double& Matrix::operator()(Args... args) {
    int indices[] = {args...};
    int index = 0;
    int multiplier = 1;
    for (int i = 0; i < dimLen; i++) {
        index += indices[i] * multiplier;
        multiplier *= dims[i];
    }
    return data[index];
}

#endif