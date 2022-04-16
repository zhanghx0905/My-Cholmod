# cholmod 接口文档

### `cholesky`

```python
def cholesky(A, beta=0, mode="auto", ordering_method="default") -> Factor
```

返回一个表示 $A+\beta I$ Cholesky 分解的 `Factor` 对象。其中 $A$ 是稀疏对称正定矩阵，$\beta$ 为标量，$I$ 是单位矩阵。

参数：

- A: `scipy.sparse` 对象。

- beta: 浮点变量，通常取 0 或 1.

- mode: supernodal 策略，有如下几种，

  ```python
  "simplicial"
  "supernodal"
  "auto"			# 由 CHOLMOD 根据矩阵特征自动选择
  ```

- ordering_method: Fill-Reducing 排序方法，有如下几种，

  ```python
  "natural"
  "amd"
  "metis"
  "nesdis"
  "colamd"
  "default"
  "best"
  ```

分解时只使用 $A$ 的下半部分。如果 $A$ 不是正定矩阵，会发出警告。 

### `cholesky_AAt`

```python
def cholesky_AAt(A, beta=0, mode="auto", ordering_method="default") -> Factor
```

返回一个表示 $AA^T+\beta I$ Cholesky 分解的 `Factor` 对象。其中 $A$ 是稀疏矩阵，$\beta$ 为标量，$I$ 是单位矩阵。

参数含义与 `cholesky` 相同。

### Class `Factor`

#### `P`

```python
def P(self) -> np.ndarray
```

返回 Fill-reducing 排列 $p$.


#### `L_D`

```python
def L_D(self) -> (sparse.csc_matrix, sparse.dia_matrix)
```

返回下三角矩阵 $L$ 和对角矩阵 $D$，使得
$$
LDL^T=A[\text{p[:, None]}, \text{p[None, :]}]=P^T A P
$$

#### `L`

```python
def L(self) -> sparse.csc_matrix
```

返回下三角矩阵 $L$ 使得
$$
LL^T=A[\text{P[:, None]}, \text{P[None, :]}] = P^T A P
$$


#### `copy`

```python
def copy(self) -> Factor
```

返回当前 `Factor` 的一个深拷贝。

### qr

```python
def qr(A: sparse.csc_matrix) \
    -> Tuple[sparse.csc_matrix, sparse.csc_matrix, np.ndarray, int]
```

$$
QR = AP
$$