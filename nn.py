# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 11:47:24 2020

@author: Laksh
"""
################################################################################
#
# LOGISTICS
#
#    Lakshmeesha Shetty
#    Netid: LSS180005
#
# DESCRIPTION
#
#    MNIST image classification with an xNN written and trained in Python
#
# INSTRUCTIONS
#
#    1. Go to Google Colaboratory: https://colab.research.google.com/notebooks/welcome.ipynb
#    2. File - New Python 3 notebook
#    3. Cut and paste this file into the cell (feel free to divide into multiple cells)
#    4. Runtime - Run all
#
# NOTES
#
#    1. This does not use PyTorch, TensorFlow or any other xNN library
#
#    2. Include a short summary here in nn.py of what you did for the neural
#       network portion of code
#
#    3. Include a short summary here in cnn.py of what you did for the
#       convolutional neural network portion of code
#
#    4. Include a short summary here in extra.py of what you did for the extra
#       portion of code
#
################################################################################

################################################################################
#
# IMPORT
#
################################################################################

#
# you should not need any import beyond the below
# PyTorch, TensorFlow, ... is not allowed
#

import os.path
import urllib.request
import gzip
import math
import time
import numpy             as np
import matplotlib.pyplot as plt


################################################################################
#
# PARAMETERS
#
################################################################################

#
# add other hyper parameters here with some logical organization
#

# data
DATA_NUM_TRAIN         = 60000
DATA_NUM_TEST          = 10000
DATA_CHANNELS          = 1
DATA_ROWS              = 28
DATA_COLS              = 28
DATA_CLASSES           = 10
DATA_URL_TRAIN_DATA    = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
DATA_URL_TRAIN_LABELS  = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
DATA_URL_TEST_DATA     = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
DATA_URL_TEST_LABELS   = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
DATA_FILE_TRAIN_DATA   = 'train_data.gz'
DATA_FILE_TRAIN_LABELS = 'train_labels.gz'
DATA_FILE_TEST_DATA    = 'test_data.gz'
DATA_FILE_TEST_LABELS  = 'test_labels.gz'
epoch = 15

# display
DISPLAY_ROWS   = 8
DISPLAY_COLS   = 4
DISPLAY_COL_IN = 10
DISPLAY_ROW_IN = 25
DISPLAY_NUM    = DISPLAY_ROWS*DISPLAY_COLS

################################################################################
#
# DATA
#
################################################################################

# download
if (os.path.exists(DATA_FILE_TRAIN_DATA)   == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_DATA,   DATA_FILE_TRAIN_DATA)
if (os.path.exists(DATA_FILE_TRAIN_LABELS) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_LABELS, DATA_FILE_TRAIN_LABELS)
if (os.path.exists(DATA_FILE_TEST_DATA)    == False):
    urllib.request.urlretrieve(DATA_URL_TEST_DATA,    DATA_FILE_TEST_DATA)
if (os.path.exists(DATA_FILE_TEST_LABELS)  == False):
    urllib.request.urlretrieve(DATA_URL_TEST_LABELS,  DATA_FILE_TEST_LABELS)

# training data
# unzip the file, skip the header, read the rest into a buffer and format to NCHW
file_train_data   = gzip.open(DATA_FILE_TRAIN_DATA, 'r')
file_train_data.read(16)
buffer_train_data = file_train_data.read(DATA_NUM_TRAIN*DATA_ROWS*DATA_COLS)
train_data        = np.frombuffer(buffer_train_data, dtype=np.uint8).astype(np.float32)
train_data        = train_data.reshape(DATA_NUM_TRAIN, 1, DATA_ROWS, DATA_COLS)

# training labels
# unzip the file, skip the header, read the rest into a buffer and format to a vector
file_train_labels   = gzip.open(DATA_FILE_TRAIN_LABELS, 'r')
file_train_labels.read(8)
buffer_train_labels = file_train_labels.read(DATA_NUM_TRAIN)
train_labels        = np.frombuffer(buffer_train_labels, dtype=np.uint8).astype(np.int32)

# testing data
# unzip the file, skip the header, read the rest into a buffer and format to NCHW
file_test_data   = gzip.open(DATA_FILE_TEST_DATA, 'r')
file_test_data.read(16)
buffer_test_data = file_test_data.read(DATA_NUM_TEST*DATA_ROWS*DATA_COLS)
test_data        = np.frombuffer(buffer_test_data, dtype=np.uint8).astype(np.float32)
test_data        = test_data.reshape(DATA_NUM_TEST, 1, DATA_ROWS, DATA_COLS)

# testing labels
# unzip the file, skip the header, read the rest into a buffer and format to a vector
file_test_labels   = gzip.open(DATA_FILE_TEST_LABELS, 'r')
file_test_labels.read(8)
buffer_test_labels = file_test_labels.read(DATA_NUM_TEST)
test_labels        = np.frombuffer(buffer_test_labels, dtype=np.uint8).astype(np.int32)

# debug
# print(train_data.shape)   # (60000, 1, 28, 28)
# print(train_labels.shape) # (60000,)
# print(test_data.shape)    # (10000, 1, 28, 28)
# print(test_labels.shape)  # (10000,)

################################################################################


# #Important Layer Data


# Initial Weights
Weight_init = np.random.uniform(-0.5,0.5,size = (784,1000))
Weight_hidl1_hidl2 = np.random.uniform(-0.5,0.5,size = (1000,100))
Weight_hidl2_final = np.random.uniform(-0.5,0.5,size = (100,10))

# Initial Bias
init_bias = np.random.uniform(-0.5,0.5,size = (1,1000))
hidl1_hidl2_bias = np.random.uniform(-0.5,0.5,size = (1,100))
hidl2_final_bias = np.random.uniform(-0.5,0.5,size = (1,10))


# ReLu Activation Function
def relu_imp(x):
    if x>0:
        return x
    else:
        return 0
    
# Differentiation of Relu activation Function with respect to Net Input   
def diff_relu(x):
    if x>0:
        return 1
    else:
        return 0
    
# Softmax Function   
def softmax(x):
    e_x = np.exp(x)
    e_y = e_x.sum(axis=1)
    return e_x/e_y
    





epoch_l = []
accuracy_l = []
loss_l = []
time_l = []

for ep in range(0,epoch):
    #Setting the Learning rate
    alpha = 0.01
    
    error = 0
    start_time = time.time()
    for j in range(0,len(train_data)):
        
        
        #one hot pmf vector
        one_hot_pmf_0_character = int(train_labels[j])
        one_hot_pmf_vector = np.zeros((1,10),dtype=float)
        one_hot_pmf_vector[0][one_hot_pmf_0_character] = float(1)
        
        
        input_temp = train_data[j,:,:,:]
        #division by 255
        input_temp_divid = input_temp/255
        #vectorization of input data
        vectorized_input_temp_flat = input_temp_divid.flatten()
        vectorized_input_temp_vect = vectorized_input_temp_flat.reshape(1,784)
        
        
        
        #Matrix multiplication from init layer to hidl1
        matrix_mult_init_result = np.dot(vectorized_input_temp_vect,Weight_init)
        #Matrix Addition by a bias from init layer to hidl1
        matrix_mult_init_result_bias = matrix_mult_init_result + init_bias
        #ReLu Activation on hidl1
        result_hidl1 = np.zeros((1,1000),dtype=float)
        for i in range(0,1000):
            result_hidl1[0][i] = relu_imp(matrix_mult_init_result_bias[0][i])
        
        
        
        #Matrix multiplication from hidl1 to hidl2
        matrix_multi_hidl1_hidl2 = np.dot(result_hidl1,Weight_hidl1_hidl2)
        #Matrix Addition by a bias from init layer to hidl1
        matrix_multi_hidl1_hidl2_bias = matrix_multi_hidl1_hidl2 + hidl1_hidl2_bias
        #ReLu Activation on hidl2
        result_hidl2 = np.zeros((1,100),dtype=float)
        for i in range(0,100):
            result_hidl2[0][i] = relu_imp(matrix_multi_hidl1_hidl2_bias[0][i])
        
        
        
        #Matrix multiplication from hidl2 to final layer
        matrix_multi_hidl2_final = np.dot(result_hidl2,Weight_hidl2_final)
        #Matrix Addition by a bias from init layer to hidl1
        matrix_multi_hidl2_final_bias = matrix_multi_hidl2_final + hidl2_final_bias
        #Applying softmax to the obtained result
        result_final = softmax(matrix_multi_hidl2_final_bias)
        
        
        

        #Loss Calculation
        temp_error = 0
        for i in range(0,len(result_final[0])):
            log_term = math.log2(result_final[0][i])
            temp_error += one_hot_pmf_vector[0][i]*log_term
        error += -1*temp_error
        print("Loss for Sample "+str(j)+" and Epoch "+str(ep)+" = "+str(-1*temp_error))
        

        
        #E with respect to Wh2_f and hidl2_final_bias
        
        
        #First step: Creating a (actual-predicted) vector
        a_p = result_final-one_hot_pmf_vector
        # Second Step: Creating E_r_hidl2_final_bias
        E_r_hidl2_final_bias = a_p
        #Final step: Creating E_r_Wh2_f matrix
        E_r_Wh2_f = np.dot(result_hidl2.T,a_p)
        
                
        
                
        
        #E with respect to Wh1_h2 and hidl1_hidl2_bias
        
        #First step: Obtaining the diff of Relu for h1_h2
        diff_relu_h1_h2 = np.zeros((1,100))
        for i in range(len(diff_relu_h1_h2[0])):
            diff_relu_h1_h2[0][i] = diff_relu(matrix_multi_hidl1_hidl2_bias[0][i])
        #Second Step: Creating E_r_hidl1_hidl2_bias
        E_r_hidl1_hidl2_bias = np.multiply(np.dot(a_p,Weight_hidl2_final.T),diff_relu_h1_h2)
        #Final step: Creating E_r_Wh1_h2 matrix
        E_r_Wh1_h2 = np.dot(result_hidl1.T,np.multiply(diff_relu_h1_h2,np.dot(a_p,Weight_hidl2_final.T)))
        
                
        
        
        
         
        #E with respect to Wint_h1 and init_bias
                
        #First Step : Obtaining the diff of Relu for init_h1
        diff_relu_init_h1 = np.zeros((1,1000))
        for i in range(len(diff_relu_init_h1[0])):
            diff_relu_init_h1[0][i] = diff_relu(matrix_mult_init_result_bias[0][i])
        #Second Step: Creating E_r_init_bias
        E_r_init_bias = np.multiply( np.dot( np.multiply( diff_relu_h1_h2, np.dot( a_p, Weight_hidl2_final.T ) ), Weight_hidl1_hidl2.T ), diff_relu_init_h1 )
        #Final step: Creating E_r_Winit_h1
        E_r_Winit_h1 = np.dot( vectorized_input_temp_vect.T, np.multiply( np.dot( np.multiply( diff_relu_h1_h2, np.dot( a_p, Weight_hidl2_final.T ) ), Weight_hidl1_hidl2.T ), diff_relu_init_h1 ) )
                
        
        
        
            
        #Weight updation
        Weight_init = Weight_init - alpha*E_r_Winit_h1
        Weight_hidl1_hidl2 = Weight_hidl1_hidl2 - alpha*E_r_Wh1_h2
        Weight_hidl2_final = Weight_hidl2_final - alpha*E_r_Wh2_f
        
        #Bias updation
        init_bias = init_bias - alpha*E_r_init_bias
        hidl1_hidl2_bias = hidl1_hidl2_bias - alpha*E_r_hidl1_hidl2_bias
        hidl2_final_bias = hidl2_final_bias - alpha*E_r_hidl2_final_bias
        
        
            
    count = 0
    predicted_array = np.zeros((0,0),dtype=int)
    for j in range(0,len(test_data)):
        
        
        one_hot_pmf_0_character = int(test_labels[j])
        one_hot_pmf_vector = np.zeros((1,10),dtype=float)
        one_hot_pmf_vector[0][one_hot_pmf_0_character] = float(1)
        
        
        input_temp = test_data[j,:,:,:]
        #division by 255
        input_temp_divid = input_temp/255
        #vectorization of input data
        vectorized_input_temp_flat = input_temp_divid.flatten()
        vectorized_input_temp_vect = vectorized_input_temp_flat.reshape(1,784)
        
        
        #Matrix multiplication from init layer to hidl1
        matrix_mult_init_result = np.dot(vectorized_input_temp_vect,Weight_init)
        #Matrix Addition by a bias from init layer to hidl1
        matrix_mult_init_result_bias = matrix_mult_init_result + init_bias
        #ReLu Activation on hidl1
        result_hidl1 = np.zeros((1,1000),dtype=float)
        for i in range(0,1000):
            result_hidl1[0][i] = relu_imp(matrix_mult_init_result_bias[0][i])
        
        
        #Matrix multiplication from hidl1 to hidl2
        matrix_multi_hidl1_hidl2 = np.dot(result_hidl1,Weight_hidl1_hidl2)
        #Matrix Addition by a bias from init layer to hidl1
        matrix_multi_hidl1_hidl2_bias = matrix_multi_hidl1_hidl2 + hidl1_hidl2_bias
        #ReLu Activation on hidl2
        result_hidl2 = np.zeros((1,100),dtype=float)
        for i in range(0,100):
            result_hidl2[0][i] = relu_imp(matrix_multi_hidl1_hidl2_bias[0][i])
        
        
        #Matrix multiplication from hidl2 to final layer
        matrix_multi_hidl2_final = np.dot(result_hidl2,Weight_hidl2_final)
        #Matrix Addition by a bias from init layer to hidl1
        matrix_multi_hidl2_final_bias = matrix_multi_hidl2_final + hidl2_final_bias
        #Applying softmax to the obtained result
        result_final = softmax(matrix_multi_hidl2_final_bias)
        
        print("Predicted: "+str(int(np.argmax(result_final,axis=1)))+" Actual: "+str(int(np.argmax(one_hot_pmf_vector,axis=1))))
        
        predicted_array = np.append(predicted_array,int(np.argmax(result_final,axis=1)))
        
        
        if int(np.argmax(result_final,axis=1))==int(np.argmax(one_hot_pmf_vector,axis=1)):
            count +=1
        
    print("##########################################################################################")
    print("Epoch: "+str(ep+1))
    acc = count/len(test_data)
    print("%s seconds" % (time.time() - start_time))
    time_l.append(time.time()-start_time)
    print("Testing Accuracy: "+str(acc*100))
    epoch_l.append(ep+1)
    accuracy_l.append(acc*100)
    loss_l.append(error/len(train_data))
    print("Average Loss in this epoch: "+str(error/len(train_data)))
    print("##########################################################################################")
    
    
print("The Highest Accuracy Reached for the Testing data: "+str(max(accuracy_l))+"%")


#Graph of accuracy vs epoch
plt.plot(epoch_l,accuracy_l,label="Testing Accuracy")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy %")
plt.title("Graph of accuracy vs epoch")
plt.legend()
plt.show()

loss_l[:] = [x*100 for x in loss_l]

#Graph of loss vs epoch
plt.plot(epoch_l,loss_l,label="Training loss")
plt.xlabel("Number of Epochs")
plt.ylabel("loss %")
plt.title("Graph of Loss vs epoch")
plt.legend()
plt.show()



#Performance display:

print("Total time taken to complete 15 epochs:"+str(sum(time_l)/(60))+" minutes")


type_l = ["Type","Input Layer","Hidden Layer 1","Hidden Layer 2", "Output Layer"]
input_size_l = ["Input Size","1 X 28 X 28","1 X 784","1 X 1000","1 X 100"]
output_size_l = ["Output Size","1 X 784","1 X 1000","1 X 100","1 X 10"]
MACs_l = ["MACs","N/A","784000","100000","1000"]

            

a = [type_l,input_size_l,output_size_l,MACs_l]
l = [len(max(i, key=len)) for i in zip(*a)]

for item in a:
    for idx in range(len(l)):
        print(item[idx].ljust(l[idx]), end='  ')
    print()
















################################################################################

#
# feel free to split this into some number of classes, functions, ... if it
# helps with code organization; for example, you may want to create a class for
# each of your layers that store parameters, performs initialization and
# includes forward and backward functions
#

# cycle through the epochs

    # set the learning rate

    # cycle through the training data
        # forward pass
        # loss
        # back prop
        # weight update

    # cycle through the testing data
        # forward pass
        # accuracy

    # per epoch display (epoch, time, training loss, testing accuracy, ...)

################################################################################
#
# DISPLAY
#
################################################################################

#
# more code for you to write
#

# accuracy display
# final value
# plot of accuracy vs epoch

# performance display
# total time
# per layer info (type, input size, output size, parameter size, MACs, ...)

# example display
# replace the xNN predicted label with the label predicted by the network
fig = plt.figure(figsize=(DISPLAY_COL_IN, DISPLAY_ROW_IN))
ax  = []
for i in range(DISPLAY_NUM):
    img = test_data[i, :, :, :].reshape((DATA_ROWS, DATA_COLS))
    ax.append(fig.add_subplot(DISPLAY_ROWS, DISPLAY_COLS, i + 1))
    ax[-1].set_title('True: ' + str(test_labels[i]) + ' xNN: ' + str(predicted_array[i]))
    plt.imshow(img, cmap='Greys')
plt.show()
