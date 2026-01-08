import random as rand


class Matrix:
    def __init__(self):
        ...
    
    @classmethod
    def randn(cls, rows, cols):
        mat = [[None]* cols] * rows
        for row in range(rows):
            for col in range(cols):
                mat[row][col] = rand.random()

        return mat

    @classmethod
    def zeros(cls, rows, cols):
        res = [[None]*cols]*rows

        for row in range(rows):
            for col in range(cols):
                res[row][col] = 0

        return res

    def matmul(self, mat1, mat2):
        res = [[None]*len(mat2[0])] * len(mat1)

        if len(mat1[0]) != len(mat2):
            return False
        
        for i in range(len(mat1)):
            mat1_row = [mat1[i][k] for k in range(len(mat1[i]))]
            for j in range(len(mat2[0])):
                mat2_col = [mat2[j][i] for i in range(len(mat2[j]))]
                
                dot_prod = 0
                for elem1, elem2 in zip(mat1_row, mat2_col):
                    print(elem1, elem2)
                    dot_prod += elem1 * elem2

                res[i][j] = dot_prod
            

        return res

mat1 = Matrix()
mat1_ = mat1.randn(2, 2)

mat2 = Matrix()
mat2_ = mat2.randn(2, 2)

print(mat1_)
print(mat2_)

res = mat1.matmul(mat1_, mat2_)

print(res)



