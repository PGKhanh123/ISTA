#ifndef ISTA_H_
#define ISTA_H_

#include <ap_fixed.h>

// System dimensions
#define M 128
#define N 256
#define MAX_ITER 200

// Fixed-point type (24-bit total, 8-bit integer)
typedef ap_fixed<24, 8,AP_RND, AP_WRAP> data_t;

// Step size MU
const data_t MU = 0.295;

// Top-level function declaration
void ista_top(
    data_t A[M][N],
    data_t y[M],
    data_t x_hat[N],
    data_t lambda
);

#endif // ISTA_H_
