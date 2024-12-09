import streamlit as st





st.logo('Image/logo.png',  size="Large", link=None, icon_image=None)
st.title('Backpropagation Algorithm')

st.sidebar.markdown("  Hi! Created with ❤️❤️❤️ by [Marvelous AiLegend](https://youtu.be/bWHBDTcrxTQ?si=bb2LgtVS2rN_BvOz).")
st.image('C:/Users/dell/OneDrive/Pictures/Screenshots/O11.png', caption = 'Backprop')

st.image('Image/robot.jpeg', caption="Backpropagation Algorithm")
st.markdown("Hey this is Aafreen Khan")
st.code("""impoort numpy as np
        import pandas as pd
        df = pd.DataFrame([[7,7.5,5],[5,6,4],[6,8,3],[4,6,25],[5,9,18],[6,8,25]], columns=['Team', 'Grade','Salary'])
        df
        def initialize_parameters(layer_dims):

  np.random.seed(3)
  parameters = {}
  L = len(layer_dims)

  for l in range(1, L):

    parameters['W' + str(l)] = np.ones((layer_dims[l-1], layer_dims[l]))*0.1
    parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))



  return parameters
  
  def linear_forward(A_prev, W, b):

  Z = np.dot(W.T, A_prev) + b

  return Z
  
  # Forward Prop
  def L_layer_forward(X, parameters):
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network

    for l in range(1, L+1):
       A_prev = A
       Wl = parameters['W' + str(l)]
       bl = parameters['b' + str(l)]
       #print("A"+str(l-1)+": ", A_prev)
     # print("W"+str(l)+": ", Wl)
     # print("b"+str(l)+": ", bl)
     # print("--"*20)
       A = linear_forward(A_prev, Wl, bl)
     # print("A"+str(l)+": ", A)
     # print("**"*20)
   return A,A_prev
   
   initialize_parameters([2,2,1])
   
   X = df[['Team', 'Grade']].values[0].reshape(2,1) # Shape(no of features, no. of training example)
   y = df[['Salary']].values[0][0]

   # Parameter initialization
   parameters = initialize_parameters([2,2,1])

   L_layer_forward(X, parameters)
   y_hat,A1 = L_layer_forward(X, parameters)
   y_hat,A1
   
   y_hat = y_hat[0][0]
   
   y_hat
   
   update_parameters(parameters,y,y_hat,A1,X)
   
   
   def update_parameters(parameters,y,y_hat,A1,X):
  parameters['W2'][0][0] = parameters['W2'][0][0] + (0.001 * 2 * (y - y_hat)*A1[0][0])
  parameters['W2'][1][0] = parameters['W2'][1][0] + (0.001 * 2 * (y - y_hat)*A1[1][0])
  parameters['b2'][0][0] = parameters['W2'][0][0] + (0.001 * 2 * (y - y_hat))

  parameters['W1'][0][0] = parameters['W1'][0][0] + (0.001 * 2 * (y - y_hat)*parameters['W2'][0][0]*X[0][0])
  parameters['W1'][0][1] = parameters['W1'][0][1] + (0.001 * 2 * (y - y_hat)*parameters['W2'][0][0]*X[1][0])
  parameters['b1'][0][0] = parameters['b1'][0][0] + (0.001 * 2 * (y - y_hat)*parameters['W2'][0][0])

  parameters['W1'][1][0] = parameters['W1'][1][0] + (0.001 * 2 * (y - y_hat)*parameters['W2'][1][0]*X[0][0])
  parameters['W1'][1][1] = parameters['W1'][1][1] + (0.001 * 2 * (y - y_hat)*parameters['W2'][1][0]*X[1][0])
  parameters['b1'][1][0] = parameters['b1'][1][0] + (0.001 * 2 * (y - y_hat)*parameters['W2'][1][0])
  
  update_parameters(parameters,y,y_hat,A1,X)
  
  X = df[['Team', 'Grade']].values[0].reshape(2,1) # Shape(no of features, no. of training example)
y = df[['Salary']].values[0][0]

# Parameter initialization
parameters = initialize_parameters([2,2,1])

y_hat,A1 = L_layer_forward(X,parameters)
y_hat = y_hat[0][0]

update_parameters(parameters,y,y_hat,A1,X)

parameters

X = df[['Team', 'Grade']].values[1].reshape(2,1) # Shape(no of features, no. of training example)
y = df[['Salary']].values[1][0]

# Parameter initialization
parameters = initialize_parameters([2,2,1])

y_hat,A1 = L_layer_forward(X,parameters)
y_hat = y_hat[0][0]

update_parameters(parameters,y,y_hat,A1,X)

parameters


X = df[['Team', 'Grade']].values[2].reshape(2,1) # Shape(no of features, no. of training example)
y = df[['Salary']].values[2][0]

# Parameter initialization
parameters = initialize_parameters([2,2,1])

y_hat,A1 = L_layer_forward(X,parameters)
y_hat = y_hat[0][0]

update_parameters(parameters,y,y_hat,A1,X)

parameters

X = df[['Team', 'Grade']].values[3].reshape(2,1) # Shape(no of features, no. of training example)
y = df[['Salary']].values[3][0]

# Parameter initialization
parameters = initialize_parameters([2,2,1])

y_hat,A1 = L_layer_forward(X,parameters)
y_hat = y_hat[0][0]

update_parameters(parameters,y,y_hat,A1,X)

parameters

X = df[['Team', 'Grade']].values[4].reshape(2,1) # Shape(no of features, no. of training example)
y = df[['Salary']].values[4][0]

# Parameter initialization
parameters = initialize_parameters([2,2,1])

y_hat,A1 = L_layer_forward(X,parameters)
y_hat = y_hat[0][0]

update_parameters(parameters,y,y_hat,A1,X)

parameters

X = df[['Team', 'Grade']].values[5].reshape(2,1) # Shape(no of features, no. of training example)
y = df[['Salary']].values[5][0]

# Parameter initialization
parameters = initialize_parameters([2,2,1])

y_hat,A1 = L_layer_forward(X,parameters)
y_hat = y_hat[0][0]

update_parameters(parameters,y,y_hat,A1,X)

parametersX = df[['Team', 'Grade']].values[5].reshape(2,1) # Shape(no of features, no. of training example)
y = df[['Salary']].values[5][0]

# Parameter initialization
parameters = initialize_parameters([2,2,1])

y_hat,A1 = L_layer_forward(X,parameters)
y_hat = y_hat[0][0]

update_parameters(parameters,y,y_hat,A1,X)

parameters


%matplotlib inline

# epochs implementation

parameters = initialize_parameters([2,2,1])
epochs = 8

for i in range(epochs):

  Loss = []

  for j in range(df.shape[0]):

    X = df[['Team', 'Grade']].values[j].reshape(2,1) # Shape(no of features, no. of training example)
    y = df[['Salary']].values[j][0]

    # Parameter initialization


    y_hat,A1 = L_layer_forward(X,parameters)
    y_hat = y_hat[0][0]

    update_parameters(parameters,y,y_hat,A1,X)

    Loss.append((y-y_hat)**2)

  print('Epoch - ',i+1,'Loss - ',np.array(Loss).mean())

parameters

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
plt.figure(figsize=(5,2))
plt.title('Model Loss Chart', color='purple')
plt.xlabel('Loss' , color='green')
plt.ylabel('epoch',color='blue')

plt.legend(['Loss', 'epoch'], loc='upper left')
plt.plot(Loss, color='darkgreen')
        """
    
)



