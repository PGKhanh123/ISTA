#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <iomanip>
#include <cmath>
#include "ista.h"

float hex2float(std::string hex_str) {
    unsigned int int_val;
    std::stringstream ss;
    ss << std::hex << hex_str;
    ss >> int_val;

    int signed_val = int_val;
    if (int_val & 0x00800000) {
        signed_val |= 0xFF000000;
    }
    
    return (float)signed_val / 65536.0f;
}

bool load_coe(const char* filepath, float* out_array, int max_size) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open " << filepath << std::endl;
        return false;
    }

    std::string line;
    bool is_data_section = false;
    int count = 0;

    while (std::getline(file, line)) {
        if (line.find("memory_initialization_vector") != std::string::npos) {
            is_data_section = true;
            continue;
        }

        if (is_data_section) {
            std::string hex_val = "";
            for (char c : line) {
                if (isxdigit(c)) {
                    hex_val += c;
                } else if (c == ',' || c == ';') {
                    if (!hex_val.empty() && count < max_size) {
                        out_array[count] = hex2float(hex_val);
                        count++;
                        hex_val = "";
                    }
                }
            }
            if (!hex_val.empty() && count < max_size) {
                out_array[count] = hex2float(hex_val);
                count++;
            }
        }
    }
    file.close();
    return true;
}

int main() {
    float A_float[M * N];
    float y_float[M];
    
    data_t A[M][N];
    data_t y[M];
    data_t x_hat[N];
    
    data_t lambda = 0.0475;

    if (!load_coe("A_24_8.coe", A_float, M * N)) return -1;
    if (!load_coe("y_24_8.coe", y_float, M)) return -1;

    int idx = 0;
    for (int i = 0; i < M; i++) {
        y[i] = y_float[i];
        for (int j = 0; j < N; j++) {
            A[i][j] = A_float[idx++];
        }
    }

    std::cout << "Starting ISTA Hardware Simulation..." << std::endl;
    ista_top(A, y, x_hat, lambda);
    std::cout << "Simulation Complete!" << std::endl;

    std::ofstream out_file("x_hat_out.txt");
    int non_zero_count = 0;
    
    for (int j = 0; j < N; j++) {
        float val = (float)x_hat[j];
        out_file << val << "\n";
        
        if (std::abs(val) > 0.1) {
            std::cout << "Pixel [" << j << "] = " << val << "\n";
            non_zero_count++;
        }
    }
    out_file.close();

    std::cout << "Total non-zero pixels: " << non_zero_count << "/256\n";

    return 0;
}
