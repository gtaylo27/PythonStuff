import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
import sys

# Number 1
data = np.genfromtxt(open("PubGData.csv", "r"), delimiter=",", skip_header=1)
print (data.shape)
print (data)

plt.scatter(data[:,3], data[:,14])
plt.show()

data_X = data[:,np.newaxis,5]
#print (data_X.shape) 
data_X_train = data_X[:-45000]
data_X_test = data_X[-5000:]
#print (data_X_train)
#print ("testbreak")
#print(data_X_test)
data_y = data[:,np.newaxis, 14]
data_y_train = data_y[:-45000]
data_y_test = data_y[-5000:]
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

#KFold 5

#five_split = cross_val_score(reg, data_X, data_y, cv=5)
five_split = KFold(n_splits = 5)
print ("five_split")
print (five_split)
count = 0
for train, test in five_split.split(data_X):
    count+=1
    #print ('train: %s, test: %s' % (train,test))
    print ("next")
    split_X_train, split_X_test = data_X[train], data_X[test]
    split_y_train, split_y_test = data_y[train], data_y[test]
    reg.fit(split_X_train, split_y_train)
    split_y_pred = reg.predict(split_X_test)
    print ("the error for fold %d is %.8f" % (count, mean_squared_error(split_y_test, split_y_pred)))
#average KFold
#five_fold_average= ((five_split[0]+five_split[1]+five_split[2]+five_split[3]+five_split[4])/5)
#print ("Error for 5 KFold plot is: %.8f" % five_fold_average)
#Multivariate Analysis
lowestError = -1;
print()
print("**********killpoints, rankpoints, winpoints**********")

print()
multi_X = data[:, (6,12,13)]
multi_X_train = multi_X[:-45000]
#print (multi_X)
multi_X_test = multi_X[-5000:]
#print (multi_X_test)
reg = LinearRegression()
multi_five_split = KFold(n_splits = 5)
count = 0
for train, test in multi_five_split.split(multi_X):
    count+=1
    #print ('train: %s, test: %s' % (train,test))
    print ("next")
    split_X_train, split_X_test = multi_X[train], multi_X[test]
    split_y_train, split_y_test = data_y[train], data_y[test]
    reg.fit(split_X_train, split_y_train)
    print ("Regression coefficients: %.9f, %.9f" % (reg.coef_[:,0],reg.coef_[:,1]))
    split_y_pred = reg.predict(split_X_test)
    print ("the error for fold %d is %.8f" % (count, mean_squared_error(split_y_test, split_y_pred)))



print()
print("**********killplace, rankpoints, winpoints**********")


print()
multi_X = data[:, (5,12,13)]
multi_X_train = multi_X[:-45000]
#print (multi_X)
multi_X_test = multi_X[-5000:]
#print (multi_X_test)
reg = LinearRegression()
multi_five_split = KFold(n_splits = 5)
count = 0
for train, test in multi_five_split.split(multi_X):
    count+=1
    #print ('train: %s, test: %s' % (train,test))
    print ("next")
    split_X_train, split_X_test = multi_X[train], multi_X[test]
    split_y_train, split_y_test = data_y[train], data_y[test]
    reg.fit(split_X_train, split_y_train)
    print ("Regression coefficients: %.9f, %.9f" % (reg.coef_[:,0],reg.coef_[:,1]))
    split_y_pred = reg.predict(split_X_test)
    print ("the error for fold %d is %.8f" % (count, mean_squared_error(split_y_test, split_y_pred)))
print()
print("**********damagedealt, headshotkills, killplace, killpoints, kills**********")

print()
multi_X = data[:, (3,4,5,6,7)]
multi_X_train = multi_X[:-45000]
#print (multi_X)
multi_X_test = multi_X[-5000:]
#print (multi_X_test)
reg = LinearRegression()
multi_five_split = KFold(n_splits = 5)
count = 0
for train, test in multi_five_split.split(multi_X):
    count+=1
    #print ('train: %s, test: %s' % (train,test))
    print ("next")
    split_X_train, split_X_test = multi_X[train], multi_X[test]
    split_y_train, split_y_test = data_y[train], data_y[test]
    reg.fit(split_X_train, split_y_train)
    print ("Regression coefficients: %.9f, %.9f" % (reg.coef_[:,0],reg.coef_[:,1]))
    split_y_pred = reg.predict(split_X_test)
    print ("the error for fold %d is %.8f" % (count, mean_squared_error(split_y_test, split_y_pred)))
    
print()
print("**********rankpoints, winpoints**********")

print()
multi_X = data[:, (12,13)]
multi_X_train = multi_X[:-45000]
#print (multi_X)
multi_X_test = multi_X[-5000:]
#print (multi_X_test)
reg = LinearRegression()
multi_five_split = KFold(n_splits = 5)
count = 0
for train, test in multi_five_split.split(multi_X):
    count+=1
    #print ('train: %s, test: %s' % (train,test))
    print ("next")
    split_X_train, split_X_test = multi_X[train], multi_X[test]
    split_y_train, split_y_test = data_y[train], data_y[test]
    reg.fit(split_X_train, split_y_train)
    print ("Regression coefficients: %.9f, %.9f" % (reg.coef_[:,0],reg.coef_[:,1]))
    split_y_pred = reg.predict(split_X_test)
    print ("the error for fold %d is %.8f" % (count, mean_squared_error(split_y_test, split_y_pred)))
    print()
print("**********killplace, killpoints**********")

print()
multi_X = data[:, (5, 6)]
multi_X_train = multi_X[:-45000]
#print (multi_X)
multi_X_test = multi_X[-5000:]
#print (multi_X_test)
reg = LinearRegression()
multi_five_split = KFold(n_splits = 5)
count = 0
for train, test in multi_five_split.split(multi_X):
    count+=1
    #print ('train: %s, test: %s' % (train,test))
    print ("next")
    split_X_train, split_X_test = multi_X[train], multi_X[test]
    split_y_train, split_y_test = data_y[train], data_y[test]
    reg.fit(split_X_train, split_y_train)
    print ("Regression coefficients: %.9f, %.9f" % (reg.coef_[:,0],reg.coef_[:,1]))
    split_y_pred = reg.predict(split_X_test)
    print ("the error for fold %d is %.8f" % (count, mean_squared_error(split_y_test, split_y_pred)))
print()
print("**********matchDuration, rankpoints, winpoints**********")

print()
multi_X = data[:, (8,12,13)]
multi_X_train = multi_X[:-45000]
#print (multi_X)
multi_X_test = multi_X[-5000:]
#print (multi_X_test)
reg = LinearRegression()
multi_five_split = KFold(n_splits = 5)
count = 0
for train, test in multi_five_split.split(multi_X):
    count+=1
    #print ('train: %s, test: %s' % (train,test))
    print ("next")
    split_X_train, split_X_test = multi_X[train], multi_X[test]
    split_y_train, split_y_test = data_y[train], data_y[test]
    reg.fit(split_X_train, split_y_train)
    print ("Regression coefficients: %.9f, %.9f" % (reg.coef_[:,0],reg.coef_[:,1]))
    split_y_pred = reg.predict(split_X_test)
    print ("the error for fold %d is %.8f" % (count, mean_squared_error(split_y_test, split_y_pred)))    
print()
print("**********matchDuration, winpoints**********")

print()
multi_X = data[:, (8,13)]
multi_X_train = multi_X[:-45000]
#print (multi_X)
multi_X_test = multi_X[-5000:]
#print (multi_X_test)
reg = LinearRegression()
multi_five_split = KFold(n_splits = 5)
count = 0
for train, test in multi_five_split.split(multi_X):
    count+=1
    #print ('train: %s, test: %s' % (train,test))
    print ("next")
    split_X_train, split_X_test = multi_X[train], multi_X[test]
    split_y_train, split_y_test = data_y[train], data_y[test]
    reg.fit(split_X_train, split_y_train)
    print ("Regression coefficients: %.9f, %.9f" % (reg.coef_[:,0],reg.coef_[:,1]))
    split_y_pred = reg.predict(split_X_test)
    print ("the error for fold %d is %.8f" % (count, mean_squared_error(split_y_test, split_y_pred)))
    print()
print("**********matchDuration, rankpoints**********")

print()
multi_X = data[:, (8,12)]
multi_X_train = multi_X[:-45000]
#print (multi_X)
multi_X_test = multi_X[-5000:]
#print (multi_X_test)
reg = LinearRegression()
multi_five_split = KFold(n_splits = 5)
count = 0
for train, test in multi_five_split.split(multi_X):
    count+=1
    #print ('train: %s, test: %s' % (train,test))
    print ("next")
    split_X_train, split_X_test = multi_X[train], multi_X[test]
    split_y_train, split_y_test = data_y[train], data_y[test]
    reg.fit(split_X_train, split_y_train)
    print ("Regression coefficients: %.9f, %.9f" % (reg.coef_[:,0],reg.coef_[:,1]))
    split_y_pred = reg.predict(split_X_test)
    print ("the error for fold %d is %.8f" % (count, mean_squared_error(split_y_test, split_y_pred)))    
print()

print("**********damagedealt, killpoints, kills**********")

print()
multi_X = data[:, (3,6,7)]
multi_X_train = multi_X[:-45000]
#print (multi_X)
multi_X_test = multi_X[-5000:]
#print (multi_X_test)
reg = LinearRegression()
multi_five_split = KFold(n_splits = 5)
count = 0
for train, test in multi_five_split.split(multi_X):
    count+=1
    #print ('train: %s, test: %s' % (train,test))
    print ("next")
    split_X_train, split_X_test = multi_X[train], multi_X[test]
    split_y_train, split_y_test = data_y[train], data_y[test]
    reg.fit(split_X_train, split_y_train)
    print ("Regression coefficients: %.9f, %.9f" % (reg.coef_[:,0],reg.coef_[:,1]))
    split_y_pred = reg.predict(split_X_test)
    print ("the error for fold %d is %.8f" % (count, mean_squared_error(split_y_test, split_y_pred)))    
print()


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