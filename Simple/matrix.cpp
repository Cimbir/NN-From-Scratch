#include "matrix.hpp"
#include <stdexcept>
#include <iostream>
#include <random>

// ================== Matrix ==================

Matrix::Matrix() {
    this->dimLen = 0;
    this->dims = nullptr;
    this->data = nullptr;
}

Matrix::Matrix(int dimLen, int* dims) {
    this->dimLen = dimLen;
    this->dims = new int[dimLen];
    int totalSize = 1;
    for (int i = 0; i < dimLen; i++) {
        this->dims[i] = dims[i];
        totalSize *= dims[i];
    }
    this->data = new double[totalSize];
    for(int i = 0; i < totalSize; i++){
        data[i] = (rand() % 1000) / 10;
    }
}

int Matrix::size(int dim) {
    if (dim < 0) {
        throw std::invalid_argument("Invalid dimension");
    } else if (dim >= dimLen) {
        return 1;
    } else {
        return dims[dim];
    }
}

void Matrix::prod(Matrix& m, Matrix& res) {
    if (dimLen < 2 && m.dimLen < 2) {
        throw std::invalid_argument("Only matrices with 2 or more dimensions are supported");
    }
    if (size(1) != m.size(0) || size(0) != res.size(0) || m.size(1) != res.size(1)) {
        throw std::invalid_argument("Invalid dimensions for matrix multiplication");
    }
    // ! ADD SUPPORT FOR HIGHER DIMENSIONS
    for(int i = 0; i < size(0); i++){
        for(int j = 0; j < m.size(1); j++){
            res(i, j) = 0;
            for(int k = 0; k < size(1); k++){
                res(i, j) += (*this)(i, k) * m(k, j);
            }
        }
    }
}

double Matrix::dot(Matrix& m) {
    if (dimLen != 1 || m.dimLen != 1) {
        throw std::invalid_argument("Invalid dimensions for dot product");
    }
    double res = 0;
    for(int i = 0; i < size(0); i++){
        res += (*this)(i) * m(i);
    }
    return res;
}

string rec_to_string(double* data, int dimLen, int* dims, int gap, int index) {
    if (dimLen == 1) {
        string res = "[";
        for (int i = 0; i < dims[0]; i++) {
            res += std::to_string(data[index + i * gap]);
            if (i < dims[0] - 1) {
                res += ", ";
            }
        }
        res += "]";
        return res;
    } else {
        string res = "[";
        for (int i = 0; i < dims[0]; i++) {
            res += rec_to_string(data, dimLen - 1, dims + 1, gap * dims[0], index + i * gap);
            if (i < dims[0] - 1) {
                res += ",\n";
            }
        }
        res += "]";
        return res;
    }
}

string Matrix::to_string() {
    return rec_to_string(data, dimLen, dims, 1, 0);
}