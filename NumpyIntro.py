import numpy as np
import matplotlib.pyplot as mpl

# Number 1
data = np.genfromtxt(open("iris.csv", "r"), delimiter=",", skip_header=1)
print data.shape

# Number 2
mpl.scatter(data[:,0],data[:,1])
mpl.show()

# Number 3
data_stack = np.genfromtxt(open("iris_stack.csv", "r"), delimiter=",", skip_header=1)
print data_stack.shape

# Number 4
data = np.vstack((data,data_stack))
print data.shape

# Number 5
data_single_row = np.reshape(data, -1)
print data_single_row.shape

# Number 6
temp = np.copy(data[:,0])
data[:,0]=data[:,1]
data[:,1]=temp
mpl.scatter(data[:,0],data[:,1])
mpl.show()