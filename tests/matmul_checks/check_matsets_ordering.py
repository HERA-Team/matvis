import numpy as np

# this test ensures that we are indexing the matrix product correctly to get the unique ant pairs when using the sub-matrix method

Nant = 10

V = np.zeros((Nant, Nant))

for i in range(Nant):
    for j in range(Nant):
        V[i, j] = Nant * i + j

print(V)

pairs = np.array(
    [[0, 3], [1, 2], [2, 3], [1, 5], [6, 7], [5, 8], [6, 9], [7, 9], [8, 9]]
)
antpairs = np.copy(pairs)

for i in range(len(pairs)):
    pairs[i] = np.array(pairs[i])

print(V[pairs[:, 0], pairs[:, 1]])  # print out the unique elements of V

matsets = [
    (np.array([0, 1, 2]), np.array([2, 3, 5])),
    (np.array([5, 6, 7, 8]), np.array([7, 8, 9])),
]

mat_product = np.zeros((Nant, Nant))
out = np.zeros(len(antpairs))

for i, (ai, aj) in enumerate(matsets):
    AI, AJ = np.meshgrid(ai, aj)
    mat_product[AI, AJ] = V[AI, AJ]

for i, (ai, aj) in enumerate(antpairs):
    out[i] = mat_product[ai, aj]

print(mat_product)
print(out)  # should be the same as the second print statement
