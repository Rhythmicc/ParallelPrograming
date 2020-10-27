1. mm: do `mv` or `vm` on multiple cores; split blocks; 
2. Jacobi: 

| ..   | iter1 | iter2 | iter3 | iter4 |       |
| ---- | ----- | ----- | ----- | ----- | ----- |
| x0   | 0     | 0.7   | 1.04  | 1.073 | core1 |
| x1   | 0     | 0.8   | 1.03  | 1.094 | core2 |
| x2   | 0     | 0.8   | 0.45  | 1.374 | ..    |
| x3   | 0     | -1    | -0.2  | -0.13 | core4 |

3. `spmv`

4. Sorcy: parallel sort

   A: [3, 10, 7, 6, 2, 9, 13, 4]

   (bitomic)

   
   
   (merge) 

   | core1 | core2 | core3 | core4 |
| ----- | ----- | ----- | ----- |
   | 3, 10 | 7, 6  | 2, 9  | 13, 4 |
   
   | core1 | core2 | core3 | core4 |
   | ----- | ----- | ----- | ----- |
   | 3, 10 | 6, 7  | 2, 9  | 4, 13 |
   
   merge:
   
   | core1       | core2       |
   | ----------- | ----------- |
   | 3, 6, 7, 10 | 2, 4, 9, 13 |
   
   merge:
   
   A: [2, 3, 4, 6, 7, 9, 10, 13]
   
   (Quick Sort)
   
5. Prefix Sum

6. 





