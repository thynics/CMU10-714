#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

// how to absract part of matrix.
// float *A
// int start, int end
// int m , int n 

namespace py = pybind11;

void mat_mul(float *A, float *B, float *C, size_t M, size_t N, size_t K) 
{
    memset(C, 0, M * K * sizeof(float));  // 初始化 C 为 0
    for (size_t m = 0; m < M; ++m) {
        for (size_t k = 0; k < K; ++k) {
            for (size_t n = 0; n < N; ++n) {
                C[m * K + k] += A[m * N + n] * B[n * K + k];
            }
        }
    }
}




void transpose(float *A, float*B, size_t m, size_t n)
{
    for(size_t i = 0; i < n; ++i) {
        for(size_t j = 0; j < m; ++j) {
            B[i * m + j] = A[j * n + i];
        }
    }
}

void mat_element_wise_mul(float *A, float *B, float *C, size_t m, size_t n)
{
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++ j) {
            C[i * n + j] = A[i * n + j] * B[i * n + j];
        }
    }
}

void mat_element_wise_add(float *A, float *B, float *C, size_t m, size_t n, float sig)
{
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            C[i * n + j] = A[i * n + j] + sig * B[i * n + j];
        }
    }
}

void soft_max_norm(float *A, size_t m, size_t n)
{
    float* row_sum = new float[m]();
    for (size_t i = 0; i < m; ++i) {
        float sum = 0;
        for(size_t j = 0; j < n; ++j) {
            sum += exp(A[i * n + j]);
        }
        row_sum[i] = sum;
    }
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            A[i * n + j] = exp(A[i * n + j]) / row_sum[i];
        }
    }
    delete(row_sum);
}

void mat_element_wise_mul_num(float* A, size_t m, size_t n, float mul){
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++ j) {
            A[i * n + j] *= mul;
        }
    }
}

// y: m * 1
// res: m * n
void one_shot_and_broadcast(unsigned char* y, size_t m, size_t n, float* res) {
    memset(res, 0, m * n * sizeof(float));
    for (size_t i = 0; i < m; ++i) {
        res[i * n + int(y[i])] = 1;
    }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    float* z = (float*)malloc(batch * k * sizeof(float));
    float* Iy = (float*)malloc(batch * k * sizeof(float));
    float* minus_mat = (float*)malloc(batch * k * sizeof(float));
    float* g = (float*)malloc(n * k * sizeof(float));
    float* x_t = (float*)malloc(n * batch * sizeof(float));
    for (size_t i = 0; i < m; i += batch) {
        size_t current_batch = (i + batch > m) ? m - i : batch;
        float* x = const_cast<float*>(X + i * n); // current_batch * n
        unsigned char* Y = const_cast<unsigned char*>(y + i); // current_batch * 1
        mat_mul(x, theta, z, current_batch, n, k); // batch * k
        soft_max_norm(z, current_batch, k); // batch * k
        one_shot_and_broadcast(Y, current_batch, k, Iy); // batch * K
        mat_element_wise_add(z, Iy, minus_mat, current_batch, k, -1); // batch * k
        transpose(x, x_t, current_batch, n); // n * batch
        mat_mul(x_t, minus_mat, g, n, current_batch, k); // n * k
        mat_element_wise_add(theta, g, theta, n, k, -lr/float(current_batch));
    }
    free(z);
    free(Iy);
    free(minus_mat);
    free(g);
    free(x_t);
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}