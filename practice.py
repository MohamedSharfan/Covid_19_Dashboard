import numpy as np

#this is how to create a array
arr = np.arange(10)
print(arr)

matrix = arr.reshape(2,5)
A = matrix
print(A)
print(A.T)

#find average
mean_value = arr.mean() # or np.mean(arr)
print("Mean: ",mean_value)

dot_product = np.dot(matrix, matrix.T)
print(dot_product)

print("First row:", matrix[0])
print("Second Column:", matrix[:,1])