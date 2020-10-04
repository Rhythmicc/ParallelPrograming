#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#ifdef LIMIT_THREAD
#include <omp.h>
#endif
#include "OmpPrimitive.h"

int main(int argc, char**argv) {
    int len = atoi(argv[1]);
    struct timeval t1, t2;
    number *arr = (number *) malloc(sizeof(number) * len), sum;
    for (int i=1;i<=len;++i) arr[i-1] = i;
    gettimeofday(&t1, NULL);
    OmpReduction(arr, len, &sum);
    gettimeofday(&t2, NULL);
    printf("OmpReduction: %.3lf ms\n", (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000 );
    gettimeofday(&t1, NULL);
    OwnReduction(arr, len, &sum);
    gettimeofday(&t2, NULL);
    printf("OwnReduction: %.3lf ms\n", (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000 );
    free(arr);
    return 0;
}
