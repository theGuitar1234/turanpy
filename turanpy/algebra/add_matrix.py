def add(mat, b):
    result = [[0 for _ in range(len(mat[0]))] for _ in range(len(mat))]
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            result[i][j] = mat[i][j] + b
    return result

if __name__ == "__main__":
    pass