#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Function to perform initial array transformation
void transform_array(int input[], int intermediate[], int N) {
    for (int i = 0; i < N; i++) {
        intermediate[i] = input[i] - input[N-1-i];
    }
}

// Function to perform weight adjustment and compute final output
void compute_output(int intermediate[], int output[], int N, int E) {
    double w[N];
   
    // Initialize weights with small random values to introduce variation
    for (int i = 0; i < N; i++) {
        w[i] = (rand() % 100) / 100.0 + 0.1; // Random weights in the range [0.1, 1.1]
    }

    for (int e = 0; e < E; e++) {
        for (int i = 0; i < N; i++) {
            output[i] = (int)(intermediate[i] * w[i]); // Casting to int for simplicity
        }

        
        // Adjust weights based on the formula
        for (int i = 0; i < N; i++) {
            if (i != N-1-i) { // Avoid log(0) which is undefined
                w[i] = w[i] - log(fabs(w[i] - w[N-1-i]));
            }
        }

       
    }
}

int main() {
    int N = 10;  // size
    int E = 5;   // number of iterations
    int input[N];
    int intermediate[N];
    int output[N];
   
    // Initialize input array with a wider range of random values
    for (int i = 0; i < N; i++) {
        input[i] = rand() % 200 - 100; // Random values in the range [-100, 100]
    }

    // Print initial input array
    printf("Input array:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", input[i]);
    }
    printf("\n");

    transform_array(input, intermediate, N);
    compute_output(intermediate, output, N, E);

    // Print final output array
    printf("Final output array:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", output[i]);
    }
    printf("\n");

    return 0;
}
