import random as rand


class Matrix:
    def __init__(self, data):
        self.data = data
    
    def __add__(self, other):
        rows = len(self.data)
        cols = len(self.data[0])

        sum_mat = zeros(rows, cols)

        for row in range(rows):
            for col in range(cols):
                sum_mat[row][col] = self.data[row][col] + other
        
        return Matrix(sum_mat)

    def __repr__(self):
        return f"Matrix({self.data})"
    
    def __getitem__(self, *keys):
        keys = keys[0]

        if isinstance(keys, int):
            return self.data[keys]
        else:
            key1, key2 = keys
            return self.data[key1][key2]


def randn(rows, cols):
    mat = [[] for _ in range(rows)]
    for row in range(rows):
        for _ in range(cols):
            mat[row].append(rand.random())

    return mat

def zeros(rows, cols):
    res = [[] for _ in range(rows)]

    for row in range(rows):
        for _ in range(cols):
            res[row].append(0)

    return res

def matmul(mat1, mat2):
    assert len(mat1[0]) == len(mat2)

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
        
    return res


class Linear:
    def __init__(self, in_channels, out_channels, bias=True):
        self.w = randn(in_channels, out_channels)

        if bias:
            self.b = 0
        else:
            self.b = None
    
    def __call__(self, x):
        out = Matrix(matmul(x, self.w)) + self.b
        
        return out
