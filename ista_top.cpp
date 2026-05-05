#include "ista.h"

void ista_top(
    data_t A[M][N],
    data_t y[M],
    data_t x_hat[N],
    data_t lambda
) {
    // Vivado HLS Interface Pragmas
    #pragma HLS INTERFACE ap_ctrl_hs port=return
    #pragma HLS INTERFACE ap_none port=lambda
    #pragma HLS INTERFACE ap_memory port=A
    #pragma HLS INTERFACE ap_memory port=y
    #pragma HLS INTERFACE ap_memory port=x_hat

    // Local buffers for internal parallel processing
    data_t local_y[M];
    data_t local_A[M][N];
    data_t local_x_hat[N];

    // Intermediate buffers
    data_t r[M]; 
    data_t g[N]; 

    // Array Partitioning for parallel access
    #pragma HLS ARRAY_PARTITION variable=local_A cyclic factor=8 dim=2
    #pragma HLS ARRAY_PARTITION variable=local_x_hat cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=g cyclic factor=8 dim=1

    // -----------------------------------------------------------------
    // Step 1: Copy data from external memory to local buffers
    // -----------------------------------------------------------------
    copy_y: for (int i = 0; i < M; i++) {
        #pragma HLS PIPELINE II=1
        local_y[i] = y[i];
    }

    copy_A_outer: for (int i = 0; i < M; i++) {
        copy_A_inner: for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            local_A[i][j] = A[i][j];
        }
    }

    init_x: for (int j = 0; j < N; j++) {
        #pragma HLS PIPELINE II=1
        local_x_hat[j] = 0;
    }

    // -----------------------------------------------------------------
    // Step 2: Main loop for ISTA algorithm
    // -----------------------------------------------------------------
    main_loop: for (int iter = 0; iter < MAX_ITER; iter++) {
        
    	// Step 2.1: Calculate residual r = y - A * x_hat
    	        res_outer: for (int i = 0; i < M; i++) {

    	            // 1. Declare an array of 8 parallel accumulators and fully partition it into individual registers
    	            data_t sum_part[8];
    	            #pragma HLS ARRAY_PARTITION variable=sum_part complete dim=1

    	            // Initialize accumulators to 0
    	            init_sum: for (int k = 0; k < 8; k++) {
    	                #pragma HLS UNROLL
    	                sum_part[k] = 0;
    	            }

    	            // 2. MAC Loop: Iterate with a stride of 8
    	            res_inner: for (int j = 0; j < N; j += 8) {
    	                #pragma HLS PIPELINE II=1

    	                // Unroll 8 multiplications, accumulating each into an independent register
    	                res_mac: for (int k = 0; k < 8; k++) {
    	                    #pragma HLS UNROLL
    	                    sum_part[k] += (data_t)local_A[i][j + k] * (data_t)local_x_hat[j + k];
    	                }
    	            }

    	            // 3. Combine the 8 branches into a single final sum (Reduction step)
    	            data_t total_sum = 0;
    	            res_reduction: for (int k = 0; k < 8; k++) {
    	                #pragma HLS UNROLL
    	                total_sum += sum_part[k];
    	            }

    	            // 4. Cast back to data_t (this automatically applies AP_RND and AP_SAT configurations)
    	            r[i] = local_y[i] - (data_t)total_sum;
    	        }

        // Step 2.2: Calculate gradient g = A^T * r
        init_g: for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            g[j] = 0;
        }

        grad_outer: for (int i = 0; i < M; i++) {
            grad_inner: for (int j = 0; j < N; j++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS UNROLL factor=8
                g[j] += local_A[i][j] * r[i];
            }
        }


        // Step 2.3: Update and apply soft-thresholding
        data_t thresh = MU * lambda;
        update_loop: for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
			#pragma HLS UNROLL factor=8
            data_t x_tmp = local_x_hat[j] + MU * g[j];
            

            if (x_tmp > thresh) {
                local_x_hat[j] = x_tmp - thresh;
            } else if (x_tmp < -thresh) {
                local_x_hat[j] = x_tmp + thresh;
            } else {
                local_x_hat[j] = 0;
            }
        }
    }

    // -----------------------------------------------------------------
    // Step 3: Copy final results from local buffers to output memory
    // -----------------------------------------------------------------
    out_loop: for (int j = 0; j < N; j++) {
        #pragma HLS PIPELINE II=1
        x_hat[j] = local_x_hat[j];
    }
}
