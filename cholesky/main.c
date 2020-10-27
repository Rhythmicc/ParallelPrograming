#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <zconf.h>
#include <omp.h>

#define VALUE_TYPE float

VALUE_TYPE A[8000 * 8000];

/// Check the result
void Check(int n, VALUE_TYPE *L) {
    int res[n];
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        res[i] = 0;
        for (int j = 0; j <= i; ++j)
            if (L[i * n + j] != 1 && L[i * n + j] != 0) {
                res[i] = 1;
                break;
            }
    }
    for (int i = 0; i < n; ++i)
        if (res[i]) {
            printf("       Matrix factorization failed.\n");
            return;
        }
    printf("       Matrix factorization succeeded.\n");
}

void cholesky(VALUE_TYPE *Raw, int n) {
    int j, k, i, cal_j, cal_i;
    VALUE_TYPE sum;

    for (j = 0; j < n; ++j) {
        sum = 0, cal_j = j * n;

#pragma omp parallel for default(none) shared(Raw, n, j, cal_j) private(k) reduction(+: sum)
        for (k = 0; k < j; ++k) sum += Raw[cal_j + k] * Raw[cal_j + k];
        
        Raw[j * n + j] = sqrt(Raw[j * n + j] - sum);

#pragma omp parallel for default(none) private(i, k, sum, cal_i) shared(j, n, Raw, cal_j) if(j < n - 1)
        for (i = j + 1; i < n; ++i) {
            sum = 0, cal_i = i * n;
            for (k = 0; k < j; ++k) sum += Raw[cal_i + k] * Raw[cal_j + k];
            Raw[cal_i + j] = (Raw[cal_i + j] - sum) / Raw[cal_j + j];
        }
    }
}

int main(int argc, const char * argv[]) {
    omp_set_num_threads(atoi(argv[1]));
    struct timeval start, end;
    int b = atoi(argv[2]);
    printf("       ******************** %d*%d Matrix *****************\n", b, b);

#pragma omp parallel for shared(A)
    for (int i = 0; i < b; ++i) for (int j = 0; j < b; ++j) A[i * b + j] = j <= i ? j + 1 : i + 1;

    /// Do cholesky factorization
    gettimeofday(&start, NULL);
    cholesky(A, b);
    gettimeofday(&end, NULL);

    double all_time = 1000 * (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000;
    printf("       the time of Cholesky factorization is %.2lf ms\n", all_time);

    Check(b, A);
    return 0;
}
