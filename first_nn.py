import numpy as np

#defining a sigmoid function 
def sigmoid(x, derive=False):
	if derive == True :
		return derivative(x) #this is used during back-propogation
	else:	
		return 1/(1+np.exp(-x))

#we use the delta function during back-propogation
#the derivative of the sigmoid function = sigmoid(1-sigmoid)
def derivative(x):
	return x*(1-x)

#inputs 
X = np.array([[0,0,1],
			[0,1,1],
			[1,0,1],
			[1,1,1]])
#outputs
Y = np.array([[0],[1],[1],[0]])

#setting seed for random no.s 
np.random.seed(2)

#initialize the network using random values 
w0 = 2*np.random.random((3,4)) - 1 
w1 = 2*np.random.random((4,1,)) -1

#training step 
steps = 100000
for i in range(0,steps):
	l0 = X 
	l1 = sigmoid(np.dot(l0, w0)) #first layer 
	l2 = sigmoid(np.dot(l1, w1)) #second layer or Yhat output 

	error = Y - l2 # cost function that has to be minimized

	#printing the error at every 10,000 steps 
	if(i%10000 == 0 ):
		our_error = "Error" + str(np.mean(np.abs(error)))
		print(our_error)

#backpropogation 
	l2_delta = error*sigmoid(l2,derive=True)
	l1_error = l2_delta.dot(w1.T)

	l1_delta = l1_error*sigmoid(l1,derive=True)

	#update weights using gradient descent 
	w1 += l1.T.dot(l2_delta)
	w0 += l0.T.dot(l1_delta)

print("Output")
print(l2)






