#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import os
import csv
import time
# Analysis

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
# Tensorflow-Lite
import tflite_runtime.interpreter as tflite
# Memory
import psutil


# In[2]:


# Path to the TensorFlow Lite model
#tflite_model_path = "Saved_models/BasicAE_P-3_150.tflite"
tflite_model_path = r"C:\Users\Realt\Documents\Model_Testing\Saved_models\BasicAE_P-3_150.tflite"

ModelName = 'BasicAE_P-3_150'
filename = ModelName + '.h5'
filename, extension = os.path.splitext(filename)
if filename:
    print("Selected file:", filename)
else:
    print("No file was selected.")


# In[3]:


parts = filename.split('_')
print(parts)


# In[4]:


# Folder
model_folder = 'Saved_models/'
# Model
model_name = parts[0]
# Selected Channel
filename = parts[1]
# Feature Vector size
window_size = int(parts[2])


# In[5]:


# ### Anomaly definition
# This cell defines the stride of the window and the percentage of the window containing anomalies for it to be considered anomalous

# Number of datapoints to itterate in window generation
stride = 1
# Anomaly Window Threshold percentage
threshold = 0.7


# In[6]:

# Before loading large datasets
print("Memory Usage before loading datasets:", psutil.virtual_memory().percent)

# ### Data Path
# Defines the path to the data file containing train and test data

path_train = r"D:\Model_Testing\data\train\P-3.npy"
path_test = r"D:\Model_Testing\data\test\P-3.npy"

#path_train = './data/train/' + filename + '.npy'
#path_test = './data/test/' + filename + '.npy'

s1 = os.path.getsize(path_train)
s2 = os.path.getsize(path_test)

data_train = np.load(path_train)
data_test = np.load(path_test)


# After loading datasets
print("Memory Usage after loading datasets:", psutil.virtual_memory().percent)

# In[7]:


# ### Loading Anomaly List
# References the labeled_anomalies.csv file for where the anomalies are located

#data = pd.read_csv (r'./labeled_anomalies.csv')
data = pd.read_csv(r"C:\Users\Realt\Documents\Model_Testing\labeled_anomalies.csv")
# Defining chan_id as index
data.set_index('chan_id', inplace = True)

# Data is loaded in as a string and must be converted to an array
filedata = data.loc[filename, 'anomaly_sequences']
if isinstance(filedata, str) == True:
    test = filedata.split(',')
else:
    test = filedata[0].split(',')

test[1].strip("[]")

counter = 0
anomaly_ranges = []
temp = []
for n in range(0, len(test)):
    temp.append(int(test[n].strip(' [] ')))
    counter += 1
    if counter == 2:
        anomaly_ranges.append(temp)
        temp = []
        counter = 0

print("Total number of labelled anomalies in dataset:", len(anomaly_ranges))
print("Ranges of anomalies: ", anomaly_ranges)


# In[8]:


### List Formation
# The train and test datasets are formed into lists to be used later

ydata = []
for x in range (0, len(data_train)):
    ydata.append(data_train[x][0])
    
testdata = []
for x in range (0, len(data_test)):
    testdata.append(data_test[x][0])


# In[9]:


# ### Rescaling
# The original data is shaped 1 to -1, from previous work, scaling the data from 1 to 0 yields better results. They are then converted to numpy arrays

X_train = np.array(ydata).reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_norm = scaler.fit_transform(X_train)
X_train_1D = X_train_norm.ravel()

X_test = np.array(testdata).reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
X_test_norm = scaler.fit_transform(X_test)
X_test_1D = X_test_norm.ravel()


# In[10]:


points_train = np.array(X_train_1D)
points_test = np.array(X_test_1D)


# In[11]:


# ### Anomaly Ranges set as list
# The anomaly ranges shown earlier are loaded as simple strings and need to be converted to referencable lists

test_list = []
print('Anomaly Ranges: ')
for n in range(0, len(anomaly_ranges)):
    test_list.append(anomaly_ranges[n][1] - anomaly_ranges[n][0])
    print(anomaly_ranges[n])
    
if (any(test_list) < window_size) == False:
    window_size = min(test_list) * threshold
    print('Resizing windows due to anomaly size\n New window size: ', window_size)


# In[12]:


# ### X_train Definition
# X_train is now loaded as an array of datapoints for the length of the train dataset, with the window size number of points at each position. The array will look like (length_X_train, window_size)

X_train = []
temp = []
for y in range(0, len(ydata)-window_size, stride):
    end = window_size + y
    for x in range(y, end):
        temp.append(ydata[x])
    X_train.append(temp)
    temp = []


# In[ ]:


# ### X_test & y_test Definition
# X_test & y_test is now loaded as an array of datapoints for the length of the train dataset, with the window size number of points at each position. The array will look like (length_X_test, window_size), (length_y_test, 1)

#model = keras.models.load_model(model_folder + model_name + '_' + filename + "_" + str(window_size) + '.tflite')
#model.summary()

# Before loading TensorFlow Lite model
print("Memory Usage before loading TensorFlow Lite model:", psutil.virtual_memory().percent)


X_val = X_train[:int(len(X_train)/20)]

# Check CPU usage
cpu_usage = psutil.cpu_percent()
print("CPU Usage:", cpu_usage)

# Check memory usage
mem_usage = psutil.virtual_memory().percent
print("Memory Usage:", mem_usage)

print("Pre tflite loading")

# Load the TensorFlow Lite model as a byte array
with open(tflite_model_path, 'rb') as f:
    model_data = f.read()

# Create TensorFlow Lite interpreter from the model byte array
#interpreter = tf.lite.Interpreter(model_content=model_data)
#interpreter.allocate_tensors()

# Load the TensorFlow Lite model
interpreter = tflite.Interpreter(model_path=tflite_model_path)
#interpreter = tflite.Interpreter(model_content=model_data)
interpreter.allocate_tensors()
print("Interpreter initialization successful")



# After loading TensorFlow Lite model
print("Memory Usage after loading TensorFlow Lite model:", psutil.virtual_memory().percent)

# Get input and output details
input_details = interpreter.get_input_details()
# Get output details
output_details = interpreter.get_output_details()
print("Input and output details passed")

# Convert X_train to numpy array
X_train_np = np.array(X_train)
print("X_train to numpy array")

# Reshape the first sample in X_train_np to (1, 150)
X_val_reshaped = X_train_np[:int(len(X_train_np)/20)][0].reshape(1, -1)[:,:150].astype(np.float32)
print("X_val reshaped")

# Perform inference
interpreter.set_tensor(input_details[0]['index'], X_val)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Inference performed")

print("X_val shape:", X_val.shape)

# Print model input details
print("\nInput details:")
for detail in input_details:
    print(detail)

# Print model output details
print("\nOutput details:")
for detail in output_details:
    print(detail)

print('\n')


# In[2]:


X_test = []
y_test = []
limit = window_size*threshold
temp = []

for y in range(0, len(points_test)-window_size, stride):
    end = window_size + y
    for x in range(y, end):
        temp.append(points_test[x])
    
    if len(anomaly_ranges) == 2:
        if y in range(anomaly_ranges[0][0] - int(limit), anomaly_ranges[0][1] + int(limit)) or y in range(anomaly_ranges[1][0] - int(limit), anomaly_ranges[1][1] + int(limit)):
            y_test.append(1)
        else:
            y_test.append(0)
    if len(anomaly_ranges) == 3:
        if y in range(anomaly_ranges[0][0] - int(limit), anomaly_ranges[0][1] + int(limit)) or y in range(anomaly_ranges[1][0] - int(limit), anomaly_ranges[1][1] + int(limit)) or y in range(anomaly_ranges[2][0] - int(limit), anomaly_ranges[2][1] + int(limit)):
            y_test.append(1)
        else:
            y_test.append(0)
    if len(anomaly_ranges) == 1:
        if y in range(anomaly_ranges[0][0] - int(limit), anomaly_ranges[0][1] + int(limit)):
            y_test.append(1)
        else:
            y_test.append(0)
            
    X_test.append(temp)
    temp = []


# In[ ]:


# ### Validataion Dataset Split
# A validation dataset is created from the X_train data before the training step. 20% of the data is taken out of the training dataset and both are relabeled


#X_val = X_train[:int(len(X_train)/20)]
X_train = X_train[int(len(X_train)/20):]


# In[ ]:


# ### NP Array Conversion
# All datasets are now converted to numpy arays and the shapes of each are printed for confirmation

X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)
y_test= np.array(y_test)

print('X_train Shape: ', np.shape(X_train))
print('X_val Shape: ', np.shape(X_val))
print('X_test Shape: ', np.shape(X_test))
print('y_test Shape: ', np.shape(y_test))


# In[ ]:


# ## Model Specific Pre-Processing
# Automation of the preprocessing step based on model type

def temporalize(X, y, lookback):
    output_X = []
    output_y = []
    for i in range(len(X)-lookback-1):
        t = []
        for j in range(1,lookback+1):
            # Gather past records upto the lookback period
            t.append(X[[(i+j+1)], :])
        output_X.append(t)
        output_y.append(y[i+lookback+1])
    return output_X, output_y

def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)


# In[ ]:


if model_name == 'CNNAE':
    features = 1  # Number of features (assuming one feature per time step)
    
    samples = X_train.shape[0]  # Number of samples
    time_steps = X_train.shape[1]  # Number of time steps
    X_train = X_train.reshape(samples, time_steps, features)

    samples = X_test.shape[0]  # Number of samples
    time_steps = X_test.shape[1]  # Number of time steps
    X_test = X_test.reshape(samples, time_steps, features)
    
    samples = X_val.shape[0]  # Number of samples
    time_steps = X_val.shape[1]  # Number of time steps
    X_val = X_val.reshape(samples, time_steps, features)
    
if model_name == 'HybridAE' or model_name == 'LSTMAE':
    timesteps = window_size
    n_features = window_size

    X, y = temporalize(X = X_train, y = np.zeros(len(X_train)), lookback = timesteps)

    X = np.array(X)
    X_train = X.reshape(X.shape[0], timesteps, n_features)
    
    X_t, y_t = temporalize(X = X_test, y = np.zeros(len(X_test)), lookback = timesteps)

    X_t = np.array(X_t)
    X_test = X_t.reshape(X_t.shape[0], timesteps, n_features)
    
    X_v, y_v = temporalize(X = X_val, y = np.zeros(len(X_val)), lookback = timesteps)

    X_v = np.array(X_v)
    X_val = X_v.reshape(X_v.shape[0], timesteps, n_features)


# In[ ]:


# ### Validation Dataset Reconstruction Error
# Calculates validation dataset reconstruction error and saves as recon_err_val

# # Post Processing Script
# Loads saved model file from the definitions at the beginning and prints a model summary to confirm model is loaded correctly

# In[ ]:


X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)
y_test= np.array(y_test)

print('X_train Shape: ', np.shape(X_train))
print('X_val Shape: ', np.shape(X_val))
print('X_test Shape: ', np.shape(X_test))
print('y_test Shape: ', np.shape(y_test))


# In[ ]:


#X_recon_val = model.predict(X_val)
# Define an empty list to store reconstruction errors for validation data
recon_err_val = []

# Iterate through each sample index in X_val
for sample_index in range(len(X_val)):
    # Access the sample using the index
    sample = X_val[sample_index]
    
    # Convert the sample to float32
    sample = sample.astype(np.float32)
    
    # Reshape the sample to (1, -1) and select the first 150 elements
    sample_reshaped = sample.reshape(1, -1)[:,:150]
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], sample_reshaped)
    
    # Invoke the interpreter
    interpreter.invoke()
    
    # Get output tensor
    output_data_val = interpreter.get_tensor(output_details[0]['index'])
    
    # Calculate the reconstruction error for the current sample
    recon_err_sample = np.mean(np.power(sample - output_data_val.reshape(sample.shape), 2))
    
    # Append the reconstruction error for the current sample to recon_err_val
    recon_err_val.append(recon_err_sample)

# Convert recon_err_val to a numpy array after the loop
recon_err_val = np.array(recon_err_val)


# In[ ]:


#X_recon_train = model.predict(X_train)
# Define an empty list to store reconstruction errors for training data
recon_err_train = []

# Iterate through each sample index in X_train
for sample_index in range(len(X_train)):
    # Access the sample using the index
    sample = X_train[sample_index]
    
    # Convert the sample to float32
    sample = sample.astype(np.float32)
    
    # Reshape the sample to (1, -1) and select the first 150 elements
    sample_reshaped = sample.reshape(1, -1)[:,:150]
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], sample_reshaped)
    
    # Invoke the interpreter
    interpreter.invoke()
    
    # Get output tensor
    output_data_train = interpreter.get_tensor(output_details[0]['index'])
    
    # Calculate the reconstruction error for the current sample
    recon_err_sample = np.mean(np.power(sample - output_data_train.reshape(sample.shape), 2))
    
    # Append the reconstruction error for the current sample to recon_err_train
    recon_err_train.append(recon_err_sample)

# Convert recon_err_train to a numpy array after the loop
recon_err_train = np.array(recon_err_train)


# In[ ]:


# ### Training Dataset Reconstruction Error
# Calculates training dataset reconstruction error and saves as recon_err_train

# ### Test Dataset Reconstruction Error
# Calculates Test dataset reconstruction error and saves as recon_err_test

# In[ ]:

start = time.time()

# Define the empty list to store reconstructed data
recon_err_test = []

# Iterate through each element in the first dimension of X_test
for i in range(X_test.shape[0]):
    # Get the sample from X_test and reshape it to (1, 150)
    sample = X_test[i][0].reshape(1, -1)[:,:150]
    
    # Convert the sample to float32
    sample = sample.astype(np.float32)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], sample)
    
    # Invoke the interpreter
    interpreter.invoke()
    
    # Get output tensor
    output_data_test = interpreter.get_tensor(output_details[0]['index'])
    
    # Append the output data to the recon_err_test list
    recon_err_test.append(output_data_test)

end = time.time()

total = end - start

ips = len(X_test)/total # inference per sec on test set

#recon_err_test = np.mean(np.power(X_test - X_recon, 2), axis=1)
#recon_err_test = np.mean(np.power(X_test - output_data_test, 2), axis=1)
#recon_err_test = np.mean(np.power(X_test - output_data_test.reshape(X_test.shape), 2), axis=1)
recon_err_test = np.array(recon_err_test)

print('\nX_train Shape: ', np.shape(X_train))
print('X_val Shape: ', np.shape(X_val))
print('X_test Shape: ', np.shape(X_test))
print('y_test Shape: ', np.shape(y_test))


# In[ ]:


# ## Model Specific Post-processing

if model_name == 'HybridAE' or model_name == 'LSTMAE':
    result = []
    for e in recon_err_test:
        result.append(np.mean(e))
    recon_err_test = result
    zero = np.zeros(int((np.shape(X_train[1])[0]))+1)
    recon_err_test = np.concatenate((zero, recon_err_test))


# In[ ]:


# ### Reconstruction Error Threshold
# Calculates the reconstruction error threshold to apply to test set predictions. If the value of the reconstruction error exceed the threshold, it is considered anomalous.

percentile = np.percentile(recon_err_test, 99.99)
thresholds = np.linspace(np.min(recon_err_test), percentile, 100)


# In[ ]:


f1s = []
for threshold in thresholds:
    y_pred = (recon_err_test > threshold).astype(int)
    f1s.append(f1_score(y_test, y_pred))

max_f1_index = np.argmax(f1s)
recon_threshold = thresholds[max_f1_index]


# In[ ]:


# ### y_pred declaration
# Declares y_pred based on where the value exceeds the threshold

y_pred = []
for x in recon_err_test:
    y_hat = [1 if x > recon_threshold else 0]
    y_pred.append(y_hat)


# In[ ]:


fpr, tpr, _ = roc_curve(y_test, recon_err_test)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 5))
lw = 1
plt.plot(fpr, tpr,
         lw=lw, label='CNN Autoencoder (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Analysis')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


# ### Confusion Matrix
# 
# This cell gives the total results of the model after the threshold has been applied

conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

fig, ax = plt.subplots(figsize=(8, 8))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

print('ROC Area: %.3f' % roc_auc_score(y_test, recon_err_test))
print('F1 Score: %.3f' % f1_score(y_test, y_pred))
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('Inference per Second: ', ips)

