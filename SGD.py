import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#### Part (a): Generating x_i, y_i training samples

num_samples=input("Please enter the train and test size: ")
num_samples=int(num_samples)
epochs=input("Enter the number of epochs: ")
epochs=int(epochs)

mean = (0, 0, 0, 0)
cov = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
x = np.random.multivariate_normal(mean, cov, num_samples)
x=np.transpose(x)
mean = [0]
cov = [[1/4]]
z = np.random.multivariate_normal(mean, cov, num_samples)
z=np.transpose(z)

theta_0=np.full((1, 1), 2, dtype=int)
theta=[[1],[1/2],[1/4],[1/8]]
theta_transpose=np.transpose(theta)

y=np.dot(theta_transpose, x) + z + theta_0

#### Part (b): SGD classifier

alpha = 0.0001 #Step size
batch_size=10
# epochs = 100 #No. of epoch
iterations=int(num_samples/batch_size) # n / batch
m = num_samples #No. of data points
np.random.seed(123) #Set the seed
theta_transpose = np.random.rand(1,4) #Pick some random values to start with
theta_00=np.random.rand(1,1)

#GRADIENT DESCENT
def gradient_descent(X, Y, theta_0, theta, epochs, alpha):
	past_errors = []
	past_thetas = [theta]

	
	for i in range(epochs):
		print("Epoch No:", i)

		x=X
		y=Y

		for j in range(iterations):
			# print("Iteration No:", j)
			rand_ind=np.random.randint(0,len(y))
			x_new=x[:, rand_ind: rand_ind+batch_size]
			x=np.delete(x, range(rand_ind, rand_ind + batch_size), axis=1)
			y_new=y[:, rand_ind: rand_ind+batch_size]
			y=np.delete(y, range(rand_ind, rand_ind + batch_size), axis=1)
			
			prediction = np.dot(theta, x_new) + theta_0
			error = prediction - y_new
			cost = 1/(2*batch_size) * np.dot(error, error.T)
			theta = theta - (alpha * (1/batch_size) * np.dot(error, x_new.T))

			theta_0= theta_0 - alpha* np.sum(error)
			# if(j==0):
			past_errors.append(cost[0])            
			past_thetas.append(theta)
		
	return theta_0, past_thetas, past_errors

#Pass the relevant variables to the function and get the new values back...
updated_theta_0, past_thetas, past_errors = gradient_descent(x, y, theta_00, theta_transpose, epochs, alpha)
updated_theta = past_thetas[-1]

#Plot the cost function...
plt.title('LMS Error vs No of iterations')
plt.xlabel('No. of iterations ( epochs * batches')
plt.ylabel('LMS Error')
plt.plot(past_errors)
plt.show()


#### Part (c): Testing

test_num_samples=num_samples
mean = (0, 0, 0, 0)
cov = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
x_test = np.random.multivariate_normal(mean, cov, test_num_samples)
x_test=np.transpose(x_test)

mean = [0]
cov = [[1/4]]
z_test = np.random.multivariate_normal(mean, cov, test_num_samples)
z_test=np.transpose(z_test)

theta=[[1],[1/2],[1/4],[1/8]]
theta_transpose=np.transpose(theta)
y_test=np.dot(theta_transpose, x_test) + z_test + theta_0

def MSE(x_test, y_test, z_test, updated_theta, updated_theta_0):
	

	h=np.dot(updated_theta, x_test) + z_test + updated_theta_0
	

	error=h-y_test

	mse = 1/(2*num_samples) * np.dot(error, error.T)

	return mse

print("MSE :", MSE(x_test,y_test, z_test, updated_theta, updated_theta_0))

####Part (d): Run it for m=10




#### Hyperparameters: Alpha=0.0001, Epochs=100 
#### MSE for m=10000: 0.20
#### MSE for m=10: 0.81

#### For less training samples, the loses do not converge and training as well as testing error are high