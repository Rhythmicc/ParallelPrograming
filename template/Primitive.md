
## Reduction

```c
void Reduction(void*array, int eleSize, int len, void*res) {
    if (sizeof(float) == eleSize) {
        float *a = (float *)array, *r = (float *)res;
        for(int i=0;i<len;++i)*r += a[i];
    } else if(sizeof(double) == eleSize) {
        double *a = (double *)array, *r = (double *)res;
        for(int i=0;i<len;++i)*r += a[i];
    }
}
///
```

## PrefixSum

```c
void PrefixSum(void*array, int eleSize, int len) {
    if (sizeof(float) == eleSize) {
        float *a = (float *)array;
        for(int i=1;i<len;++i) a[i] += a[i-1];
    } else if(sizeof(double) == eleSize) {
        double *a = (double *)array;
        for(int i=1;i<len;++i) a[i] += a[i-1];
    }
}
///
```
