import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# Number 1
data = np.genfromtxt(open("PubgData.csv", "r"), delimiter=",", skip_header=1)
# print data.shape

# Number 2
data_X = data[:,np.newaxis,5:7]
data_Y = data[:,np.newaxis,14]

data_X_train = data_X[:-25000]
data_X_test = data_X[25000:]

data_Y_train = data_Y[:-25000]
data_Y_test = data_Y[25000:]

# Number 3 & 4
reg = linear_model.LinearRegression()

reg.fit(data_X_train, data_Y_train)

data_Y_pred = reg.predict(data_X_test)

# Number 5
plt.scatter(data_X_test, data_Y_test,  color='black')
plt.plot(data_X_test, data_Y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

# Number 6
print ("Mean squared error: %.12f" % mean_squared_error(data_Y_test, data_Y_pred))

# Number 7
five_split = cross_val_score(reg, data_X, data_Y , cv=5)
print(five_split)

mean_five_split_error = (five_split[0] + five_split[1] + five_split[2] + five_split[3] + five_split[4])/5
print("Mean error of the cross val: %.12f" % mean_five_split_error)

# Number 8
    # a.
 

    # b.