
import LinearModels as lm
import numpy as np


# Parameters for data generation
num_samples = 1000  # Number of samples
num_features = 10   # Number of features
random_seed = 42

# Set random seed
np.random.seed(random_seed)

# Generate random features (X)
X = np.random.rand(num_samples, num_features) * 10
Y = np.random.randint(0,2,num_samples)
test_x = np.random.rand(10,num_features) * 10

# The format that my logistic regression function requires is not row[column values] but column[row values], so if there are 10 features, we need 10 entries in the main list and 1000 entries within them
X = X.T

logreg = lm.LogisticRegression()

# Train data, Test data, Learning rate and epoch
print(logreg.fit(X,Y,0.01,10))
print(logreg.predict(test_x.T))