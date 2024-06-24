import pandas as pd
import numpy as np
import os
import time
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import keras

def process_file(filename):
    filename, _ = os.path.splitext(filename)
    parts = filename.split('_')
    model_folder = 'Saved_models_full/'
    model_name = parts[0]
    channel = parts[1]
    window_size = int(parts[2])
    print(filename)

    stride = 1
    threshold = 0.7

    path_train = './data/train/' + channel + '.npy'
    path_test = './data/test/' + channel + '.npy'

    data_train = np.load(path_train)
    data_test = np.load(path_test)

    data = pd.read_csv(r'./labeled_anomalies.csv')
    data.set_index('chan_id', inplace=True)

    filedata = data.loc[channel, 'anomaly_sequences']
    if isinstance(filedata, str):
        test = filedata.split(',')
    else:
        test = filedata[0].split(',')

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

    ydata = [data_train[x][0] for x in range(len(data_train))]
    testdata = [data_test[x][0] for x in range(len(data_test))]

    X_train = np.array(ydata).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_norm = scaler.fit_transform(X_train)
    X_train_1D = X_train_norm.ravel()

    X_test = np.array(testdata).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_test_norm = scaler.fit_transform(X_test)
    X_test_1D = X_test_norm.ravel()

    points_train = np.array(X_train_1D)
    points_test = np.array(X_test_1D)

    test_list = [anomaly_ranges[n][1] - anomaly_ranges[n][0] for n in range(len(anomaly_ranges))]

    if not any(test_list) < window_size:
        window_size = int(min(test_list) * threshold)

    X_train = []
    for y in range(0, len(ydata)-window_size, stride):
        end = window_size + y
        X_train.append(ydata[y:end])

    model = keras.models.load_model(model_folder + model_name + '_' + channel + "_" + str(window_size) + '.h5')

    X_test = []
    y_test = []
    limit = window_size * threshold
    for y in range(0, len(points_test) - window_size, stride):
        end = window_size + y
        temp = points_test[y:end]

        if any(y in range(a[0] - int(limit), a[1] + int(limit)) for a in anomaly_ranges):
            y_test.append(1)
        else:
            y_test.append(0)

        X_test.append(temp)

    X_val = X_train[:int(len(X_train)/20)]
    X_train = X_train[int(len(X_train)/20):]

    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
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


# In[ ]:


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
    
    if model_name == 'HybridAE' or model_name == 'LSTMAE' or model_name == 'LSTM' or model_name == 'Hybrid':
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

    X_recon_val = model.predict(X_val)
    recon_err_val = np.mean(np.power(X_val - X_recon_val, 2), axis=1)

    X_recon_train = model.predict(X_train)
    recon_err_train = np.mean(np.power(X_train - X_recon_train, 2), axis=1)

    start = time.time()
    
    print('Starting Loop')
    for i in range(5):
        X_recon = model.predict(X_test)
    
    end = time.time()

    total = end - start
    ips = len(X_test) / total  # inference per sec on test set

    recon_err_test = np.mean(np.power(X_test - X_recon, 2), axis=1)

    if model_name == 'HybridAE' or model_name == 'LSTMAE' or model_name == 'LSTM' or model_name == 'Hybrid':
        result = []
        for e in recon_err_test:
            result.append(np.mean(e))
        recon_err_test = result
        zero = np.zeros(int((np.shape(X_train[1])[0]))+1)
        recon_err_test = np.concatenate((zero, recon_err_test))

    percentile = np.percentile(recon_err_test, 99.99)
    thresholds = np.linspace(np.min(recon_err_test), percentile, 100)

    f1s = [f1_score(y_test, (recon_err_test > t).astype(int)) for t in thresholds]
    max_f1_index = np.argmax(f1s)
    recon_threshold = thresholds[max_f1_index]

    y_pred = [(1 if x > recon_threshold else 0) for x in recon_err_test]

    print(f'ROC Area: {roc_auc_score(y_test, recon_err_test):.3f}')
    print(f'F1 Score: {f1_score(y_test, y_pred):.3f}')
    print(f'Inference per Second: {ips}')
    print('----------------------------------------------------------------------------\n')

def main():
    print('----------------------------------------------------------------------------')
    model_folder = 'Saved_models_full/'
    files = [f for f in os.listdir(model_folder) if f.endswith('.h5')]

    for filename in files:
        process_file(filename)
        #print(filename)

if __name__ == "__main__":
    main()
    print('All Models Tested')
