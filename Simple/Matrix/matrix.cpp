#include "matrix.hpp"
#include <stdexcept>
#include <iostream>
#include <random>

// ================== Matrix ==================

Matrix::Matrix() {
    this->n = 0;
    this->m = 0;
    this->data = nullptr;
}

Matrix::Matrix(int n) {
    this->n = n;
    this->m = 1;
    this->data = new double[n];
    for(int i = 0; i < n; i++){
        data[i] = (double)(rand() % 1000) / 100;
    }
}

Matrix::Matrix(int n, int m) {
    this->n = n;
    this->m = m;
    this->data = new double[n * m];
    for(int i = 0; i < n * m; i++){
        data[i] = (double)(rand() % 1000) / 100;
    }
}

double& Matrix::operator()(int i) {
    if (i < 0 || i >= n) {
        throw std::invalid_argument("Invalid index");
    }
    return data[i];
}

double& Matrix::operator()(int i, int j) {
    if (i < 0 || i >= n || j < 0 || j >= m) {
        throw std::invalid_argument("Invalid index");
    }
    return data[i * m + j];
}

int Matrix::size(int dim) {
    if (dim == 0) {
        return n;
    } else if (dim == 1) {
        return m;
    } else {
        throw std::invalid_argument("Invalid dimension");
    }
}

void Matrix::add(Matrix& m, Matrix& res, double alpha) {
    if (n != m.size(0) || m.size(1) != res.size(1) || n != res.size(0)) {
        throw std::invalid_argument("Invalid dimensions for matrix addition");
    }
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m.size(1); j++){
            res(i, j) = (*this)(i, j) + m(i, j) * alpha;
        }
    }
}

void Matrix::mult(Matrix& m, Matrix& res) {
    if (n != m.size(0) || m.size(1) != res.size(1) || n != res.size(0)) {
        throw std::invalid_argument("Invalid dimensions for matrix multiplication");
    }
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m.size(1); j++){
            // cout << res(i, j) << " ";
            // cout << (*this)(i, j) << endl;
            res(i, j) = (*this)(i, j) * m(i, j);
        }
    }
}

void Matrix::prod(Matrix& m, Matrix& res, int transpose) {
    int fT = (transpose&2)>>1;
    int sT = transpose&1;
    if (size(1-fT) != m.size(0+sT) || size(0+fT) != res.size(0) || m.size(1-sT) != res.size(1)) {
        throw std::invalid_argument("Invalid dimensions for matrix multiplication");
    }
    for(int i = 0; i < size(0+fT); i++){
        for(int j = 0; j < m.size(1-sT); j++){
            res(i, j) = 0;
            for(int k = 0; k < size(1-fT); k++){
                
                if(fT & sT) res(i, j) += (*this)(k, i) * m(j, k);
                else if(fT) res(i, j) += (*this)(k, i) * m(k, j);
                else if(sT) res(i, j) += (*this)(i, k) * m(j, k);
                else        res(i, j) += (*this)(i, k) * m(k, j);

            }
        }
    }
}

double Matrix::dot(Matrix& m) {
    if (size(1) != 1 || m.size(1) != 1 || size(0) != m.size(0)) {
        throw std::invalid_argument("Invalid dimensions for dot product");
    }
    double res = 0;
    for(int i = 0; i < size(0); i++){
        res += (*this)(i) * m(i);
    }
    return res;
}

double Matrix::dot(Matrix& m, int dim, int index) {
    if (m.size(1) != 1) {
        throw std::invalid_argument("Invalid dimensions for matrix dot product");
    }
    if (dim < 0 || dim >= 2) {
        throw std::invalid_argument("Invalid dimension");
    }
    if (size(dim) != m.size(0)) {
        throw std::invalid_argument("Invalid dimensions for dot product");
    }
    if (index < 0 || index >= size(1 - dim)) {
        throw std::invalid_argument("Invalid index");
    }
    double res = 0;
    if (dim == 0) {
        for(int i = 0; i < size(0); i++){
            res += (*this)(i, index) * m(i);
        }
    } else {
        for(int i = 0; i < size(1); i++){
            res += (*this)(index, i) * m(i);
        }
    }
    return res;
}

string Matrix::to_string() {
    string res = "[";
    for(int i = 0; i < n; i++){
        res += "[";
        for(int j = 0; j < m; j++){
            res += std::to_string((*this)(i, j));
            if(j < m - 1) res += ", ";
        }
        res += "]";
        if(i < n - 1) res += ", ";
    }
    res += "]";
    return res;
}