#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <string>
#include <stdexcept>

using namespace std;

class Matrix{
private:
    int n;
    int m; 
    double* data;
    
public:
    Matrix();
    Matrix(int n);
    Matrix(int n, int m);

    double& operator()(int i);
    double& operator()(int i, int j);

    int size(int dim);
    void add(Matrix& m, Matrix& res, double alpha = 1);
    void mult(Matrix& m, Matrix& res);
    void prod(Matrix& m, Matrix& res, int transpose = 0);
    
    double dot(Matrix& m);
    double dot(Matrix& m, int dim, int index);

    string to_string();
};

#endif