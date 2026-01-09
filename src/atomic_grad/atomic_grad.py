import random as rand
from math import e, log


class Matrix:
    def __init__(self, data):
        self.data = data
        self.grad = zeros_like(self)
        self._backward = lambda x: None
        self._prev = []

        if not isinstance(self.data[0], (float, int)):
            self.shape = (len(self.data), len(self.data[0]))
        else:
            self.shape = (len(self.data),)


    def squeeze(self):
        assert len(self.data) == 1 or len(self.data[0]) == 1
        
        res = []
        for i in range(len(self.data)):
            for j in range(len(self.data[0])):
                res.append(self.data[i][j])

        return Matrix(res)
    
    def tolist(self):
        res = []

        for elem in self.data:
            res.append(elem)
        
        return res

    def __add__(self, other):
        rows = len(self.data)
        cols = len(self.data[0])

        sum_mat = zeros(rows, cols)

        for row in range(rows):
            for col in range(cols):
                sum_mat[row][col] = self.data[row][col] + other
        
        out = sum_mat

        out._prev = [self]

        def _backward():
            for i in range(rows):
                for j in range(cols):
                    self.grad[i][j] += 1 * out.grad[i][j]
        
        out._backward = _backward

        return out
    
    def __repr__(self):
        return f"Matrix({self.data})"
    
    def __getitem__(self, *keys):
        keys = keys[0]

        if isinstance(keys, (int, float)):
            keys = int(keys)
            return self.data[keys]
        
        else:
            key1, key2 = keys
            return self.data[key1][key2]

    def __len__(self):
        return len(self.data)

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self.data[key] = value
        else:
            key1 = key[0]
            key2 = key[1]

            self.data[key1][key2] = value

        return self.data

def relu(x):
    parent = x
    for i in range(len(x)):
        for j in range(len(x[0])):
            elem = x[i, j]
            x[i, j] = max(0, elem)
    
    out = x

    out._prev = [parent]

    def _backward():
        for i in range(len(parent)):
            for j in range(len(parent[0])):
                grad = 0 if parent[i, j] < 0 else 1
                parent.grad[i, j] += grad * out.grad[i, j]
    
    out._backward = _backward

    return out

# def zeros_like(mat):
#     rows = len(mat)
#     cols = len(mat[0])
#     res = [[] for _ in range(rows)]

#     for row in range(rows):
#         for _ in range(cols):
#             res[row].append(0)

#     return Matrix(res)

def zeros_like(mat):
    rows, *rest = mat.shape
    cols = rest[0] if rest else 1
    res = [[] for _ in range(rows)]

    for r in range(rows):
        for _ in range(cols):
            res[r].append(0)

    return Matrix(res)

def randn(rows, cols):
    mat = [[] for _ in range(rows)]
    
    for row in range(rows):
        for _ in range(cols):
            mat[row].append(rand.random())

    return Matrix(mat)

def ones(rows, cols):
    res = [[] for _ in range(rows)]

    for row in range(rows):
        for _ in range(cols):
            res[row].append(1)
    
    return Matrix(res)

def matrix(x): return Matrix(x)

def zeros(rows, cols):
    res = [[] for _ in range(rows)]

    for row in range(rows):
        for _ in range(cols):
            res[row].append(0)

    return Matrix(res)

def matmul(mat1, mat2):
    assert len(mat1[0]) == len(mat2)

    X = mat1
    W = mat2

    res_rows = len(mat1)
    res_cols = len(mat2[0])
    res = zeros(res_rows, res_cols)
    
    for i in range(len(mat1)):
        mat1_row = [mat1[i][k] for k in range(len(mat1[i]))]

        for j in range(len(mat2[0])):
            mat2_col = [mat2[n][j] for n in range(len(mat2))]
            dot_prod = 0

            for elem1, elem2 in zip(mat1_row, mat2_col):
                dot_prod += (elem1 * elem2)

            res[i][j] = dot_prod
        
    out = res
    
    out._prev = [X, W]

    def _backward():
        for i in range(len(X)):
            for j in range(len(W[0])):
                for k in range(len(W)):
                    X.grad[i, k] += out.grad[i, j] * W[k, j]
                    W.grad[k, j] += out.grad[i, j] * X[i, k]

    out._backward = _backward

    return out

def exp(mat):
    rows = len(mat)
    cols = len(mat[0])

    for i in range(rows):
        for j in range(cols):
            mat[i, j] = e**mat[i, j]
    
    return mat

def normalize(mat):
    rows = len(mat)
    cols = len(mat[0])
    res = zeros_like(mat)
    for i in range(rows):
        row_sum = sum(mat[i])
        for j in range(cols):
            res[i, j] = mat[i, j] / row_sum
    
    return res

def softmax(x):
    exp_nums = exp(x)
    norms = normalize(exp_nums)

    return norms

def cross_entropy_loss(x, target):
    logits = x
    probs = softmax(logits)
    probs = probs.squeeze()

    target = target.squeeze()
    targ_class = target[0]

    out = -log(probs[targ_class])

    out._prev = [x]

    def _backward():
        for i in range(len(logits[0])):
            grad = probs[i]
            if i == targ_class:
                grad -= 1

            x.grad[0, i] += grad * out.grad

    out._backward = _backward

    return out
    


class Linear:
    def __init__(self, in_channels, out_channels):
        self.w = randn(in_channels, out_channels)

    def __call__(self, x):
        out = matmul(x, self.w)
        
        return out


class Softmax:
    def __call__(self, x):
        norms = softmax(x)

        return norms


class CELoss:
    def __call__(self, x, target):
        loss = cross_entropy_loss(x, target)

        return loss


def backward(loss):
    topo = []
    visited = set()

    def build(v):
        if v in visited:
            return
        visited.add(v)
        
        for child in v._prev:
            build(child)

        topo.append(v)
    
    build(loss)

    loss.grad = zeros_like(loss)
    loss.grad[0, 0] = 1

    for node in reversed(topo):
        node._backward()