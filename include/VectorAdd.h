void VectorAdd(void*va, void*vb, void*vc, int eleSize, int len) {
    if (sizeof(float) == eleSize) {
        float *a = (float *) va, *b = (float *) vb, *c = (float *) vc;
        for (int i = 0; i < len; ++i) c[i] = a[i] + b[i];
    } else if (sizeof(double) == eleSize) {
        double *a = (double *) va, *b = (double *) vb, *c = (double *) vc;
        for (int i = 0; i < len; ++i) c[i] = a[i] + b[i];
    }
}