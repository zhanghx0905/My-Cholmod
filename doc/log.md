#### 环境配置

新装了一个 WSL ubuntu 20.04.

[ubuntu | 镜像站使用帮助 | 清华大学开源软件镜像站 | Tsinghua Open Source Mirror](https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/)

```sh
sudo vim /etc/apt/sources.list
sudo apt-get update

sudo apt-get install libsuitesparse-dev python3-pytest python3-pip
pip3 install Cython scikit-sparse
```

装好后，CHOLMOD 的头文件在 `/usr/include/suitesparse` 下.

### 算法原理

> Fill-reducing permutation P 是为了减少分解后 L 中新增非零元素 (fill-in) 的个数。

### 核心源码

#### 常量定义

索引类型 itype

```cython
# itype defines the types of integer used:
cdef enum:
    CHOLMOD_INT, CHOLMOD_LONG
```

矩阵类型 xtype，主要使用 REAL 和 COMPLEX

```cython
# CHOLMOD_PATTERN: x and z are ignored.
# CHOLMOD_DOUBLE: x is non-null of size nzmax, z is ignored.
# CHOLMOD_COMPLEX: x is non-null of size 2*nzmax doubles, z is ignored.
cdef enum:
    CHOLMOD_PATTERN, CHOLMOD_REAL, CHOLMOD_COMPLEX
```

数值类型 dtype

```cython
# dtype defines what the numerical type is (double or float):
cdef enum:
    CHOLMOD_DOUBLE
```

超节点策略

```cython
# supernodal strategy (for Common->supernodal)
cdef enum:
    CHOLMOD_AUTO, CHOLMOD_SIMPLICIAL, CHOLMOD_SUPERNODAL
```

错误码

```cython
# Common->status values. zero means success, negative means a fatal error,
# positive is a warning.
cdef enum:
    CHOLMOD_OK, CHOLMOD_NOT_POSDEF
    CHOLMOD_NOT_INSTALLED, CHOLMOD_OUT_OF_MEMORY, CHOLMOD_TOO_LARGE, CHOLMOD_INVALID, CHOLMOD_GPU_PROBLEM
```

排序方法

```cython
# ordering method (also used for L->ordering)
cdef enum:
    CHOLMOD_NATURAL, CHOLMOD_GIVEN, CHOLMOD_AMD, CHOLMOD_METIS						
    CHOLMOD_NESDIS, CHOLMOD_COLAMD, CHOLMOD_POSTORDERED
```

Linear System Type，用于 `solve` 传参。

> 如果不实现 solve 相关功能，似乎不必要。

```cython
cdef enum:
	CHOLMOD_A, CHOLMOD_LDLt, CHOLMOD_LD, CHOLMOD_DLt, CHOLMOD_L
    CHOLMOD_Lt, CHOLMOD_D, CHOLMOD_P, CHOLMOD_Pt
```

#### 重要 C 结构体

`cholmod_common` 配置工作参数：

```cython
ctypedef struct cholmod_common:
    int supernodal	# 超节点策略
    int status		# 错误码
    int itype, dtype	# 索引和数据类型
    int print		# 信息粒度 0 = not
    int nmethods, postorder	# 备选 method 的数量，是否 postorder
    cholmod_method_struct * method
    void (*error_handler)(int status, const char * file, int line, const char * msg)	# 错误处理函数

ctypedef struct cholmod_method_struct:
    int ordering	# 排序方法
```

> The selected ordering is followed by a weighted postorder of the elimination tree by default (see cholmod postorder for details), unless Common->postorder is set to FALSE. The postorder does not change the number of nonzeros in L or the floating-point operation count. It does improve performance, particularly for the supernodal factorization.

`cholmod_sparse` CSC 稀疏矩阵：

```cython
ctypedef struct cholmod_sparse:
    size_t nrow, ncol, nzmax
    void * p 	# column pointers
    void * i 	# row indices
    void * x	# data
    int stype 	# 0 = unsymmetric, >0 = upper triangular, <0 = lower triangular
    int itype, dtype, xtype
    int sorted	# is p sorted, 通常 = 1
    int packed	# 通常 = 1 
```

`cholmod_dense`

```cython
ctypedef struct cholmod_dense:
    size_t nrow, ncol, nzmax, d	# d is leading dimension, just set = nrow
    void * x	# nrow * ncol
    int dtype, xtype
```

`cholmod_factor`

```cython
ctypedef struct cholmod_factor:
    size_t n		# n by n
    size_t minor	# If failed, L->minor is the column at which it failed
    void * Perm		# size n, permutation used
    int itype, xtype
    # is_ll: TRUE if LL’, FALSE if LDL’
    # is_super: TRUE if supernodal, FALSE if simplicial
    # is_monotonic:  TRUE if columns of L appear in order 0..n-1.
    int is_ll, is_super, is_monotonic
    
    void * x
    # for simplicial:
    size_t nzmax  	# size of x
    void * p		# col pointers
    
    # for supernodel
    size_t nsuper	# number of supernodes
    size_t xsize	# size of x, real part of supernodes
    void * super_ "super"	# size nsuper+1, first col in each supernode
    void * pi		# size nsuper+1, pointers to integer patterns
    void * px		# size nsuper+1, pointers to real parts
```

#### 重要 C 接口

以下都是完成分解的必要接口。

```cython
int cholmod_start(cholmod_common *) except *

int cholmod_finish(cholmod_common *) except *

int cholmod_free_sparse(cholmod_sparse **, cholmod_common *) except *

int cholmod_free_dense(cholmod_dense **, cholmod_common *) except *

int cholmod_free_factor(cholmod_factor **, cholmod_common *) except *

cholmod_factor * cholmod_copy_factor(cholmod_factor *, cholmod_common *) except? NULL

cholmod_sparse * cholmod_factor_to_sparse(cholmod_factor *,
                                          cholmod_common *) except? NULL

int cholmod_change_factor(int to_xtype, int to_ll, int to_super,
                          int to_packed, int to_monotonic,
                          cholmod_factor *, cholmod_common *) except *

cholmod_factor * cholmod_analyze(cholmod_sparse *, cholmod_common *) except? NULL

# 通过 cholmod_sparse.stype 区分对 A 还是 AAt 做分解
int cholmod_factorize_p(cholmod_sparse *, double beta[2],
                        int * fset, size_t fsize,	# 设为 null 和 0 即可
                        cholmod_factor *,
                        cholmod_common *) except *
```

#### 重要 Py 接口

`cholesky` 调用 `analyze` 和 `_cholesky_inplace`, 与之相关的接口都是必要的。

`L_D` 和 `L`、 `P` 是 `Factor` 的必要功能。
