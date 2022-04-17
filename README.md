# sparseKit

Cython wrapper of SuiteSparse for sparse Cholesky/QR decomposition with almost no overhead.

[API doc](./doc/doc.md) (in Chinese)

#### API Performance Test

##### Cholesky

Tested on AMD Ryzen™ 7 5800H

| Case | number of non-zeros | Python (s) | C (s)    |
| ---- | ------------------- | ---------- | -------- |
| 1    | 77592               | 0.003041   | 0.002981 |
| 2    | 813716              | 0.471594   | 0.472786 |
| 3    | 7128473             | 132.5375   | 132.3255 |
| 4    | 18940627            | 325.8464   | 326.8807 |

##### QR

Tested on Intel Core™ i7-8750H

| Case | number of non-zeros | Python (s) | C (s)  |
| ---- | ------------------- | ---------- | ------ |
| 1    | 77592               | 0.3404     | 0.3480 |
| 2    | 106526              | 2.620      | 2.541  |

### References

[SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse)

[scikit-sparse](https://github.com/scikit-sparse/scikit-sparse)

