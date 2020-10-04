#include <string.h>
#include <immintrin.h>
#include "basic.h"

/**
 * 计算array的累加和 (omp reduction)
 * @param array 起始地址
 * @param len 数组长度
 * @param res 结果地址
 */
void OmpReduction(const number*array, int len, number*res) {
    number sum = 0;
#pragma omp parallel for reduction(+:sum)
    for(int i=0;i<len;++i)sum += array[i];
    *res = sum;
}

/**
 * 计算array的累加和 (My own algorithm)
 * @param array 起始地址
 * @param len 数组长度
 * @param res 结果地址
 */
void OwnReduction(const number*array, int len, number*res) {
    number *tmp = (number *) malloc(sizeof(number) * len);
    number *a[2] = {tmp, tmp + (len >> 1)};
    int turn = 0, init_flag = 1;
    while (len) {
        int tlen = len >> 1;
        if (init_flag) {
#pragma omp parallel for
            for (int i = 0; i < tlen; ++i) a[turn][i] = array[i << 1 | 1] + array[i << 1];
            if (len & 1) a[turn][tlen - 1] += array[len - 1];
            init_flag ^= 1;
        } else {
#pragma omp parallel for
            for (int i = 0; i < tlen; ++i) a[turn][i] = a[turn ^ 1][i << 1 | 1] + a[turn ^ 1][i << 1];
            if (len & 1) a[turn][tlen - 1] += a[turn ^ 1][len - 1];
        }
        turn ^= 1;
        len >>= 1;
    }
    *res = a[0][0] > a[1][0] ? a[0][0] : a[1][0];
    free(tmp);
}

void OmpPrefixSum(number*array, int len) {
    for(int i=1;i<len;++i) array[i] += array[i-1];
}
