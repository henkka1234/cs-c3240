#%%
#config Completer.use_jedi = False  # enable code auto-completion
import numpy as np                   # import numpy package under shorthand "np"
import pandas as pd                  # import pandas package under shorthand "pd"
import matplotlib.pyplot as plt
from nose.tools import assert_equal
from numpy.testing import assert_array_equal
import numpy as np #import numpy to work with arrays
import pandas as pd #import pandas to manipulate the dataset
from matplotlib import pyplot as plt #import the module matplotlib.pyplot to do visulization
import sklearn
from sklearn.preprocessing import PolynomialFeatures    # function to generate polynomial and interaction features
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, accuracy_score    # function to calculate mean squared error 
# Read in the data stored in the file 'FMIData_Assignment4.csv'

df = pd.read_csv('liikennetilastot.csv', sep=';')
# Print the first 5 rows of the DataFrame 'df'
for col in df.columns:
    print(col)

X = df['month'].to_numpy().reshape(-1, 1)
y = df['accidents'].to_numpy()

from sklearn.model_selection import train_test_split 

# Split the dataset into a training set and a validation set like:
# X_train, X_val, y_train, y_val = ....
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.4,random_state=42)
# YOUR CODE HERE

## define a list of values for the maximum polynomial degree 
degrees = [5, 10, 15]    

# we will use this variable to store the resulting training errors for each polynomial degree
tr_errors = []          
val_errors = []

plt.figure(figsize=(8, 20))    # create a new figure with size 8*20
for i, degree in enumerate(degrees):    # use for-loop to fit polynomial regression models with different degrees
    plt.subplot(len(degrees), 1, i + 1)    # choose the subplot
    
    lin_regr = LinearRegression(fit_intercept=False) # NOTE: "fit_intercept=False" as we already have a constant iterm in the new feature X_poly
 
    poly = PolynomialFeatures(degree=degree)    # generate polynomial features
    X_train_poly = poly.fit_transform(X_train)    # fit and transform the raw features
    lin_regr.fit(X_train_poly, y_train)
    
    # apply linear regression to these new features and labels
    
    # y_pred_train = ...    # predict values for the training data using the linear model
    y_pred_train = lin_regr.predict(X_train_poly) 
    tr_error = mean_squared_error(y_train, y_pred_train)        # calculate the training error
    X_val_poly = poly.fit_transform(X_val)  # transform the raw features for the validation data
    y_pred_val = lin_regr.predict(X_val_poly)      # predict values for the validation data using the linear model 
    val_error = mean_squared_error(y_val,y_pred_val)       # calculate the validation error
    
    # YOUR CODE HERE
    #raise NotImplementedError()
    
    # sanity check the feature matrix is transformed correctly
    assert X_val_poly.shape == (X_val.shape[0], degree + 1), "The dimension of new feature vector is incorrect" 
    # sanity check the error values
    #assert 8 < tr_error < 12 and 8 < val_error < 12
    
    tr_errors.append(tr_error)
    val_errors.append(val_error)
    X_fit = np.linspace(0, 60, 100)    # generate samples
    plt.tight_layout()
    plt.plot(X_fit, lin_regr.predict(poly.transform(X_fit.reshape(-1, 1))), label="Model")    # plot the polynomial regression model
    plt.scatter(X_train, y_train, color="b", s=10, label="Train Datapoints")    # plot a scatter plot of y(maxtmp) vs. X(mintmp) with color 'blue' and size '10'
    plt.scatter(X_val, y_val, color="r", s=10, label="Validation Datapoints")    # do the same for validation data with color 'red'
    plt.xlabel('month')    # set the label for the x/y-axis
    plt.ylabel('Number of accidents')
    plt.legend(loc="best")    # set the location of the legend
    plt.title(f'Polynomial degree = {degree}\nTraining error = {tr_error:.5}\nValidation error = {val_error:.5}')    # set the title

plt.show()    # show the plot

# sanity check the length of array tr_errors
assert len(tr_errors) == len(val_errors) == len(degrees)

plt.figure(figsize=(8, 6))

plt.plot(degrees, tr_errors, label = 'Train')
plt.plot(degrees, val_errors,label = 'Valid')
plt.legend(loc = 'upper left')

plt.xlabel('Degree')
plt.ylabel('Loss')
plt.title('Train vs validation loss')
plt.show()
# %%
