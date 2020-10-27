#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mmio.h"
#include "mmiohighlevel.h"
#include <immintrin.h>
#include <omp.h>
#include <math.h>

#define min(a, b) (a)<(b)?(a):(b)
#define max(a, b) (a)>(b)?(a):(b)
__m256 v;
int nthreads;

typedef struct {
    VALUE_TYPE *value;
    int *columnindex;
    int *rowpointer;
} SMatrix;

void scan(int *array, int n) {
    int np = ceil(1.0 * n / nthreads), old, new, *lastitem = (int *) malloc(sizeof(int) * nthreads);

#pragma omp parallel for
    for (int tid = 0; tid < nthreads; ++tid) {
        int start = tid * np, end = start + np;
        start = start > n ? n : start;
        end = end > n ? n : end;
        if (start == end) {
            lastitem[tid] = 0;
            continue;
        }
        old = array[start];
        array[start] = 0;
        for (int i = start + 1; i <= end; ++i)
            if (i != end) {
                new = array[i];
                array[i] = old + array[i - 1];
                old = new;
            } else lastitem[tid] = old + array[i - 1];
    }

    old = lastitem[0];
    lastitem[0] = 0;
    for (int i = 1; i < nthreads; ++i) {
        new = lastitem[i];
        lastitem[i] = old + lastitem[i - 1];
        old = new;
    }

#pragma omp parallel for
    for (int tid = 0; tid < nthreads; ++tid) {
        int start = tid * np, end = start + np, bias = lastitem[tid];
        start = start > n ? n : start;
        end = end > n ? n : end;
        for (int i = start; i < end; ++i) array[i] += bias;
    }
    free(lastitem);
}

void DenseToCSR(VALUE_TYPE *C, int m, int n, SMatrix *csr) {
    csr->rowpointer = (int *) malloc(sizeof(int) * (m + 1));
#pragma omp parallel for
    for(int i=0;i<=m;++i) csr->rowpointer[i] = 0;
#pragma omp parallel for
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            if (C[i * n + j]) ++csr->rowpointer[i];

    scan(csr->rowpointer, m + 1);
    csr->columnindex = (int *) malloc(sizeof(int) * (csr->rowpointer[m]));
    csr->value = (VALUE_TYPE *) malloc(sizeof(VALUE_TYPE) * csr->rowpointer[m]);
    int *row = csr->rowpointer, *col = csr->columnindex;
    VALUE_TYPE*val = csr->value;
#pragma omp parallel for
    for (int i = 0; i < m; ++i) {
        int sp_c[n], len = 0;
        VALUE_TYPE*C_line = C + i * n;
        for (int j = 0; j < n; ++j)if (C_line[j]) sp_c[len++] = j;
        for (int j = 0; j < len; ++j) {
            col[row[i] + j] = sp_c[j];
            val[row[i] + j] = C_line[sp_c[j]];
        }
    }
}

int main(int argc, char **argv) {
    struct timeval t1, t2, t3, t4;
    int size1 = 0;
    int size2 = 0;
    int *tc1;
    int *tc2;
    register double bias = -0.3000;
    nthreads = omp_get_max_threads();

    int mA, nA, nnzA, isSymmetricA;
    SMatrix A;

    int mB, nB, nnzB, isSymmetricB;
    SMatrix B[120];

    int mC, nC;

    // load matrix data from file
    gettimeofday(&t3, NULL);
    char filename1[] = "sparse-images-1024.tsv";
    mmio_info(&mA, &nA, &nnzA, &isSymmetricA, filename1);
    A.value = (VALUE_TYPE *) malloc((nnzA) * sizeof(VALUE_TYPE));
    A.columnindex = (int *) malloc((nnzA) * sizeof(int));
    A.rowpointer = (int *) malloc((mA + 1) * sizeof(int));
    mmio_data(A.rowpointer, A.columnindex, A.value, filename1);
    printf("input matrix A: ( %i, %i ) nnz = %i\n", mA, nA, nnzA);
    char neuronfile1[] = "neuron1024/n1024-l";
    char neuronfile2[] = ".tsv";
    char filename3[60];

    for (int k = 0; k < 120; k++) {
        char filenum[5];
        snprintf(filenum, sizeof(filenum), "%d", k + 1);

        strcpy(filename3, neuronfile1);
        strcat(filename3, filenum);
        strcat(filename3, neuronfile2);

        mmio_info(&mB, &nB, &nnzB, &isSymmetricB, filename3);
        B[k].value = (VALUE_TYPE *) malloc((nnzB) * sizeof(VALUE_TYPE));
        B[k].columnindex = (int *) malloc((nnzB) * sizeof(int));
        B[k].rowpointer = (int *) malloc((mB + 1) * sizeof(int));
        mmio_data(B[k].rowpointer, B[k].columnindex, B[k].value, filename3);
    }

    gettimeofday(&t4, NULL);
    double time_load = (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
    printf("Weight matrix load time: %f ms \n", time_load);

    mC = mA;
    nC = nB;
    long TolC = mA * nB;
    double sum_gemm = 0, sum_bias = 0;
    v = _mm256_setzero_ps();
    VALUE_TYPE *C0 = (VALUE_TYPE *) malloc(TolC * sizeof(VALUE_TYPE));

    gettimeofday(&t3, NULL);
    for (int k = 0; k < 120; ++k) {
#pragma omp parallel for
        for (int i = 0; i < TolC; i += 8) _mm256_storeu_ps(C0 + i, v);

        gettimeofday(&t1, NULL);
        SMatrix *CurB = B + k;
        VALUE_TYPE *B_val = CurB->value, *A_val = A.value;
        int *B_col = CurB->columnindex, *B_row = CurB->rowpointer, *A_row = A.rowpointer, *A_col = A.columnindex;
#pragma omp parallel for
        for (int i = 0; i < mA; ++i) {
            int start_j = A_row[i], end_j = A_row[i + 1], start_r, end_r;
            VALUE_TYPE val, *C_line = C0 + i * nC;
            for (int j = start_j; j < end_j; ++j) {
                start_r = B_row[A_col[j]], end_r = B_row[A_col[j] + 1], val = A_val[j];
                for (int r = start_r; r < end_r; ++r)
                    C_line[B_col[r]] += val * B_val[r];
            }
        }
        gettimeofday(&t2, NULL);
        double time_gemm = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        sum_gemm += time_gemm;

        gettimeofday(&t1, NULL);
#pragma omp parallel for
        for (int i = 0; i < TolC; ++i)
            C0[i] = min(max(C0[i] + bias, 0), 32);
        gettimeofday(&t2, NULL);
        double time_biasrelu = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        sum_bias += time_biasrelu;
        printf("k = %d, GEMM time: %4.5f ms, Bias+ReLU+Trans time: %4.5f ms\n", k + 1, time_gemm, time_biasrelu);

        if (k != 119) {
            free(A.rowpointer);
            free(A.columnindex);
            free(A.value);
            DenseToCSR(C0, mC, nC, &A);
        }
    }

    gettimeofday(&t4, NULL);
    double time_inference = (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
    printf("Inference time: %f ms; gemm: %f ms; bias: %f ms\n", time_inference, sum_gemm, sum_bias);


    // check results
    printf("test\n");
    FILE *fs;
    fs = fopen("sparse-images-1024-1.tsv", "w+");
    for (int i = 0; i < mC; ++i) {
        int sum = 0;
        for (int j = (i * nC); j < ((i + 1) * nC); ++j) sum += C0[j];
        if (sum != 0) fprintf(fs, "%d\n", i + 1);
    }
    fclose(fs);
    FILE *fp2 = NULL;

    fp2 = fopen("sparse-images-1024-1.tsv", "rb");
    if (fp2 == NULL) printf("Error:Open file fail!\n");

    fseek(fp2, 0, SEEK_END);
    size2 = ftell(fp2);
    rewind(fp2);

    tc2 = (int *) malloc(sizeof(int) * size2 / 4);

    int readnum2 = fread(tc2, 4, size2 / 4, fp2);

    fclose(fp2);

    FILE *fp1;

    fp1 = fopen("neuron1024-l120-categories.tsv", "rb");
    if (fp1 == NULL) printf("Error:Open file fail!\n");

    fseek(fp1, 0, SEEK_END);
    size1 = ftell(fp1);
    rewind(fp1);

    tc1 = (int *) malloc(sizeof(int) * size1 / 4);

    int readnum1 = fread(tc1, 4, size1 / 4, fp1);

    fclose(fp1);
    int judge = 0;
    for (int i = 0; i < size1 / 4; i++) if (tc1[i] - tc2[i] != 0) judge++;
    printf("judge:%d\n", judge);
    if (judge == 0) printf("CHALLENGE PASSED\n");
    else printf("CHALLENGE FAILED\n");

    for(int i=0;i<120;++i){
        free(B[i].rowpointer);
        free(B[i].columnindex);
        free(B[i].value);
    }

    free(A.rowpointer);
    free(A.columnindex);
    free(A.value);
    free(C0);
    return 0;
}