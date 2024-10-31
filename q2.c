#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>

#define MAX_THREADS 4

typedef struct {
    int start;
    int end;
    int *input;
    int *intermediate;
    int *output;
    int N;
    int E;
    double *w;
} thread_data_t;

void* transform_array_thread(void* arg) {
    thread_data_t *data = (thread_data_t *)arg;
    for (int i = data->start; i < data->end; i++) {
        data->intermediate[i] = data->input[i] - data->input[data->N - 1 - i];
    }
    return NULL;
}

void* compute_output_thread(void* arg) {
    thread_data_t *data = (thread_data_t *)arg;
    for (int e = 0; e < data->E; e++) {
        for (int i = data->start; i < data->end; i++) {
            data->output[i] = (int)(data->intermediate[i] * data->w[i]);
        }


        // Adjust weights based on the formula
        for (int i = data->start; i < data->end; i++) {
            if (i != data->N - 1 - i) {  // Avoid log(0) which is undefined
                data->w[i] = data->w[i] - log(fabs(data->w[i] - data->w[data->N - 1 - i]));
            }
        }

        
    }
    return NULL;
}

int main() {
    int N = 10;  // size
    int E = 5;   // number of iterations
    int input[N];
    int intermediate[N];
    int output[N];
    double w[N];
    pthread_t threads[MAX_THREADS];
    thread_data_t thread_data[MAX_THREADS];

    // Initialize input array with a wider range of random values
    for (int i = 0; i < N; i++) {
        input[i] = rand() % 200 - 100;  // Random values in the range [-100, 100]
    }

    // Initialize weights with small random values to introduce variation
    for (int i = 0; i < N; i++) {
        w[i] = (rand() % 100) / 100.0 + 0.1;  // Random weights in the range [0.1, 1.1]
    }

    // Print initial input array
    printf("Input array:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", input[i]);
    }
    printf("\n");

    int chunk_size = N / MAX_THREADS;
    for (int t = 0; t < MAX_THREADS; t++) {
        thread_data[t].start = t * chunk_size;
        thread_data[t].end = (t == MAX_THREADS - 1) ? N : (t + 1) * chunk_size;
        thread_data[t].input = input;
        thread_data[t].intermediate = intermediate;
        thread_data[t].output = output;
        thread_data[t].N = N;
        thread_data[t].E = E;
        thread_data[t].w = w;
        pthread_create(&threads[t], NULL, transform_array_thread, &thread_data[t]);
    }

    for (int t = 0; t < MAX_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }

    for (int t = 0; t < MAX_THREADS; t++) {
        pthread_create(&threads[t], NULL, compute_output_thread, &thread_data[t]);
    }

    for (int t = 0; t < MAX_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }

    // Print final output array
    printf("Final output array:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", output[i]);
    }
    printf("\n");

    return 0;
}
