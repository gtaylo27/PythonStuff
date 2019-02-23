import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# Number 1
data = np.genfromtxt(open("PubGData.csv", "r"), delimiter=",", skip_header=1)
print (data.shape)
print (data)

plt.scatter(data[:,3], data[:,14])
plt.show()

data_X = data[:,np.newaxis,5]
#print (data_X.shape) 
data_X_train = data_X[:-25000]
data_X_test = data_X[-25000:]
#print (data_X_train)
#print ("testbreak")
#print(data_X_test)
data_y = data[:,np.newaxis, 14]
data_y_train = data_y[:-25000]
data_y_test = data_y[-25000:]
#print ("data_y_train")
#print (data_y_train)
#print ("data_y_test")
#print (data_y_test)


reg = LinearRegression()
reg.fit(data_X_train, data_y_train)

data_y_pred = reg.predict(data_X_test)

plt.scatter(data_X_test, data_y_test, color="black")
plt.plot(data_X_test, data_y_pred, color="red", linewidth=2)

plt.xticks(())
plt.yticks(())

plt.show()

print ("Mean Squared Error: %.12f" % mean_squared_error(data_y_test, data_y_pred))

five_split = cross_val_score(reg, data_X, data_y, cv=5)
print(five_split)

#TEST COMMENT FROM JOEY



#data_X

# Number 2
#mpl.scatter(data[:,0],data[:,1])
#mpl.show()

# Number 3
#data_stack = np.genfromtxt(open("iris_stack.csv", "r"), delimiter=",", skip_header=1)
#print (data_stack.shape)

# Number 4
#data = np.vstack((data,data_stack))
#print (data.shape)

# Number 5
#data_single_row = np.reshape(data, -1)
#print (data_single_row.shape)

# Number 6
#temp = np.copy(data[:,0])
#data[:,0]=data[:,1]
#data[:,1]=temp
#mpl.scatter(data[:,0],data[:,1])
#mpl.show()