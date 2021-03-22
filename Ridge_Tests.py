### Loading Data
#-------------------------------------------------------------------------------
# import packages
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

# Load data from files as numpy arrays 
dollars = np.genfromtxt("Monthly_Corn_USD_per_Ton.csv", delimiter = ",", skip_header=1)
# Discard the date and index so we have only the values also "vertically stack"
# the array so it is in the correct orientation for matrix multiplications
dollars = np.vstack(dollars[:,2])
# Do the same for the other files
precipitation = np.genfromtxt("Precipitation.csv", delimiter = ",", skip_header=1)
precipitation = np.vstack(precipitation[:,2])
temperature = np.genfromtxt("Temperatures.csv", delimiter = ",", skip_header=1)
temperature = np.vstack(temperature[:,2])
#-------------------------------------------------------------------------------
### Ridge Regression based solely off precipitation
#-------------------------------------------------------------------------------
# Add a column of ones to account for the bias in the data
trainPercent = .8
numberForTrain = int(trainPercent * len(precipitation))
numberForTest = len(precipitation) - numberForTrain

trainData = precipitation[:numberForTrain]
testData = precipitation[numberForTrain:]

trainResults = dollars[:numberForTrain]
testResults = dollars[numberForTrain:]

bias = np.ones(shape = (len(trainData),1))
X = np.concatenate((bias, trainData), 1)
I = np.identity(2)
s = []
r = []

for i in range(10):
    s.append(float(i))
    # Matrix form of linear regression
    coeffs = la.inv((X.T @ X) + s[i] * I) @ X.T @ trainResults
    bias = coeffs[0][0]
    slope = coeffs[1][0]
    

    residuals = np.zeros(numberForTest)
    predictions = np.zeros(numberForTest)
    for j in range(numberForTest):
        predictions[j] = testData[j]*slope + bias
        residuals[j] = testResults[j] - predictions[j]

    MSE = np.mean(residuals**2)
    r.append(MSE)
    #print("S VALUE: "+ str(s[i]) + ", R Squared:"+ str(r[i]))

plt.plot(s,r)
plt.xlabel("Value of S")
plt.ylabel("Average of R Squared")
plt.title("Ridge Regression Least Squares for Precipitation Train:" + str(trainPercent*100) +"% Test:" + str(100-trainPercent*100)  + "% ")
plt.show()



numberForTrain = int(trainPercent * len(precipitation))
numberForTest = len(precipitation) - numberForTrain

trainData = precipitation[:numberForTrain]
testData = precipitation[numberForTrain:]

trainResults = dollars[:numberForTrain]
testResults = dollars[numberForTrain:]

bias = np.ones(shape = (len(trainData),1))
X = np.concatenate((bias, trainData), 1)


s.append(float(i))
# Matrix form of linear regression
coeffs = la.inv((X.T @ X)) @ X.T @ trainResults
bias = coeffs[0][0]
slope = coeffs[1][0]
    

residuals = np.zeros(numberForTest)
predictions = np.zeros(numberForTest)
for j in range(numberForTest):
    predictions[j] = testData[j]*slope + bias
    residuals[j] = testResults[j] - predictions[j]

MSE = np.mean(residuals**2)
print("R Squared:"+ str(MSE))

plt.plot(predictions, label="Prediction")
plt.plot(testResults, label="Results")
plt.xlabel("Date")
plt.ylabel("Price per Ton ($USD)")
plt.title("Linearly Predicted Prices From Precipitation Train:" + str(trainPercent*100) +"% Test:" + str(100-trainPercent*100)  + "% ")
plt.legend()
plt.show()

# #-------------------------------------------------------------------------------
# ### Ridge Regression based solely off Temperature
# #-------------------------------------------------------------------------------
# # Add a column of ones to account for the bias in the data
# bias = np.ones(shape = (len(temperature),1))
# X = np.concatenate((bias, temperature), 1)
# I = np.identity(2)
# s = []
# r = []

# for i in range(100):
#     s.append(float(i))
#     # Matrix form of linear regression
#     coeffs = la.inv((X.T @ X) + s[i]**2 * I) @ X.T @ dollars
#     bias = coeffs[0][0]
#     slope = coeffs[1][0]
    

#     residuals = np.zeros(len(dollars))
#     predictions = np.zeros(len(dollars))
#     for j in range(len(dollars)):
#         predictions[j] = temperature[j]*slope + bias
#         residuals[j] = dollars[j] - temperature[j]

#     MSE = np.mean(residuals**2)
#     r.append(MSE)
#     #print("S VALUE: "+ str(s[i]) + ", R Squared:"+ str(r[i]))

# plt.plot(s,r)
# plt.xlabel("Value of S")
# plt.ylabel("Summation of R Squared for " + )
# plt.title("Ridge Regression Least Squares for Temperature")
# plt.show()

# #-------------------------------------------------------------------------------
# ### Ridge Regression based solely off Temperature
# #-------------------------------------------------------------------------------
# # Add a column of ones to account for the bias in the data
# bias = np.ones(shape = (len(temperature),1))
# X = np.concatenate((bias, temperature, precipitation), 1)
# I = np.identity(3)
# s = []
# r = []

# for i in range(100):
#     s.append(float(i)/100)
#     # Matrix form of linear regression
#     coeffs = la.inv((X.T @ X) + s[i]**2 * I) @ X.T @ dollars
#     bias = coeffs[0][0]
#     tempSlope = coeffs[1][0]
#     precSlope = coeffs[2][0]
    

#     residuals = np.zeros(len(dollars))
#     predictions = np.zeros(len(dollars))
#     for j in range(len(dollars)):
#         predictions[j] = temperature[j]*tempSlope + precipitation[j]*precSlope + bias
#         residuals[j] = dollars[j] - predictions[j]

#     MSE = np.mean(residuals**2)
#     r.append(MSE)
#     #print("S VALUE: "+ str(s[i]) + ", R Squared:"+ str(r[i]))

# plt.plot(s,r)
# plt.xlabel("Value of S")
# plt.ylabel("Summation of R Squared")
# plt.title("Ridge Regression Least Squares for Both Temp and Precipitation")
# plt.show()


