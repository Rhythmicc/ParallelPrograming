
## OmpReduction

```c
void OmpReduction(void*array, int eleSize, int len, void*res) {
    if (sizeof(float) == eleSize) {
        float *a = (float *)array, *r = (float *)res;
#pragma omp for schedule(dynamic)
        for(int i=0;i<len;++i)*r += a[i];
    } else if(sizeof(double) == eleSize) {
        double *a = (double *)array, *r = (double *)res;
#pragma omp for schedule(dynamic)
        for(int i=0;i<len;++i)*r += a[i];
    }
}
///
```

## OmpPrefixSum

```c
void OmpPrefixSum(void*array, int eleSize, int len) {
    if (sizeof(float) == eleSize) {
        float *a = (float *)array;
#pragma omp for schedule(dynamic)
        for(int i=1;i<len;++i) a[i] += a[i-1];
    } else if(sizeof(double) == eleSize) {
        double *a = (double *)array;
#pragma omp for schedule(dynamic)
        for(int i=1;i<len;++i) a[i] += a[i-1];
    }
}
///
```
