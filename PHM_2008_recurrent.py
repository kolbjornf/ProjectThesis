import matplotlib
import pandas as pd
import numpy as np
import time as time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,accuracy_score

from keras.optimizers import RMSprop
from keras.models import Sequential, load_model, Input, Model
from keras.layers import Dense, LSTM, Activation, GRU, SimpleRNN #Dropout
from keras.callbacks import EarlyStopping, Callback, TensorBoard
from additional_classes import Dropout
import tqdm

import matplotlib.pyplot as plt
plt.style.use('ggplot')
#%matplotlib inline

Name = "LSTM-all-params-batch{}".format(int(time.time()))
Name2 = "LSTM-all-params-epochs{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(Name), update_freq='batch', histogram_freq=10, write_graph=True)
tensorboard2 = TensorBoard(log_dir='logs/{}'.format(Name2), update_freq='epoch', histogram_freq=10, write_graph=True)

dataset_train=pd.read_csv('train.txt',sep=' ',header=None).drop([26,27],axis=1)
col_names = ['id','cycle','setting1','setting2','setting3','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']
dataset_train.columns=col_names

dataset_test=pd.read_csv('test.txt',sep=' ',header=None).drop([26,27],axis=1)
dataset_test.columns=col_names

pm_truth=pd.read_csv('PM_truth.txt',sep=' ',header=None).drop([1],axis=1)
pm_truth.columns=['more']
pm_truth['id']=pm_truth.index+1

# generate column max for test data
rul = pd.DataFrame(dataset_test.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']

# run to failure
pm_truth['rtf']=pm_truth['more']+rul['max']
pm_truth.drop('more', axis=1, inplace=True)
dataset_test=dataset_test.merge(pm_truth,on=['id'],how='left')
dataset_test['ttf']=dataset_test['rtf'] - dataset_test['cycle']
dataset_test.drop('rtf', axis=1, inplace=True)
dataset_train['ttf'] = dataset_train.groupby(['id'])['cycle'].transform(max)-dataset_train['cycle']

df_train=dataset_train.copy()
df_test=dataset_test.copy()

period=30
#df_train['label_bc'] = df_train['ttf'].apply(lambda x: 1 if x <= period else 0)
#df_test['label_bc'] = df_test['ttf'].apply(lambda x: 1 if x <= period else 0)
    
df_train['label_bc'] = df_train['ttf']
df_test['label_bc'] = df_test['ttf']

features_col_name=['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11',
                's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
target_col_name='label_bc'
sc=MinMaxScaler()
df_train[features_col_name]=sc.fit_transform(df_train[features_col_name])
df_test[features_col_name]=sc.transform(df_test[features_col_name])

seq_length=50
seq_cols=features_col_name


def gen_sequence(id_df, seq_length, seq_cols):
    df_zeros=pd.DataFrame(np.zeros((seq_length-1,id_df.shape[1])),columns=id_df.columns)
    id_df=df_zeros.append(id_df,ignore_index=True)
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    lstm_array=[]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        lstm_array.append(data_array[start:stop, :])
    return np.array(lstm_array)

# function to generate labels
def gen_label(id_df, seq_length, seq_cols,label):
    df_zeros=pd.DataFrame(np.zeros((seq_length-1,id_df.shape[1])),columns=id_df.columns)
    id_df=df_zeros.append(id_df,ignore_index=True)
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    y_label=[]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        y_label.append(id_df[label][stop])
    return np.array(y_label)

X_train=np.concatenate(list(list(gen_sequence(df_train[df_train['id']==id], seq_length, seq_cols)) for id in df_train['id'].unique()))
y_train=np.concatenate(list(list(gen_label(df_train[df_train['id']==id], 50, seq_cols,'label_bc')) for id in df_train['id'].unique()))
X_test=np.concatenate(list(list(gen_sequence(df_test[df_test['id']==id], seq_length, seq_cols)) for id in df_test['id'].unique()))
y_test=np.concatenate(list(list(gen_label(df_test[df_test['id']==id], 50, seq_cols,'label_bc')) for id in df_test['id'].unique()))

nb_features =X_train.shape[2]
timestamp=seq_length


def prob_failure(machine_id, model):
    model_pred = model
    machine_df=df_test[df_test.id==machine_id]
    machine_test=gen_sequence(machine_df,seq_length,seq_cols)
    m_pred=model_pred.predict(machine_test)
    prob_failure = m_pred
    prob_failure=list(m_pred[-1]*100)[0]
    return prob_failure

def predict_RULs(numOfUnits, model):
    mc_predictions = np.zeros((numOfUnits, 50))
    for j in range(0,numOfUnits):
        machine_df=df_test[df_test.id==j + 1]
        machine_test=gen_sequence(machine_df,seq_length,seq_cols)
        for i in tqdm.tqdm(range(0,50)):
            y_p = model.predict(machine_test, batch_size=200)
            mc_predictions[j, i] = y_p[-1]
    pred_mean = np.zeros((numOfUnits))
    pred_std = np.zeros((numOfUnits))
    for k in range(0,numOfUnits):
        pred_mean[k] = np.mean(mc_predictions[k,:])
        pred_std[k] = np.std(mc_predictions[k,:])
    print(pred_mean)
    print(pred_std)
    plot_with_uncertainty(numOfUnits, pred_mean, pred_std)

def plot_with_uncertainty(numOfUnits, pred_mean, pred_std):
    actual_rul=np.array(pd.read_csv('PM_truth.txt',sep=' ',header=None).drop([1],axis=1))
    actual_rul = actual_rul[:numOfUnits]
    plt.plot(np.arange(numOfUnits), pred_mean, 'r-', label='Predictive mean')
    plt.plot(np.arange(numOfUnits), actual_rul, 'b', label='Training data')
    plt.fill_between(np.arange(numOfUnits), 
                    pred_mean + 2 * pred_std, 
                    pred_mean - 2 * pred_std, 
                    alpha=0.5, label='Epistemic uncertainty')
    plt.title('Prediction')
    plt.legend()
    plt.show()




def binary_classification(num_epochs, load_ex_model):
    if load_ex_model == 0:
        model = Sequential()
        model.add(SimpleRNN(
                    input_shape=(timestamp, nb_features),
                    units=100,
                    return_sequences=True, activation='tanh'))
        model.add(Dropout(0.2, training = True))
        model.add(SimpleRNN(
                    units=50,
                    return_sequences=False, activation='tanh'))
        model.add(Dropout(0.2, training = True))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=num_epochs, batch_size=200, validation_split=0.2, verbose=2, callbacks=[tensorboard, tensorboard2])
        #EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
        model.save("classification_model_saved")
    else:
        model = load_model("classification_model_saved")

    model.summary()
    scores = model.evaluate(X_train, y_train, verbose=1, batch_size=200)
    print('Accuracy: {}'.format(scores[1]))
    y_pred = model.predict_classes(X_test)
    print('Accuracy of model on test data: ',accuracy_score(y_test,y_pred))
    print('Confusion Matrix: \n',confusion_matrix(y_test,y_pred))
    for Id in range(1, 10):
        print('Probability that machine {id} will fail within {period} days: '.format(id=Id, period=period), prob_failure(Id, model))




def RUL_regression(num_epochs, numOfUnits, load_ex_model):
    if load_ex_model == 0:
        model = Sequential()
        model.add(GRU(
                    input_shape=(timestamp, nb_features),
                    units=100,
                    return_sequences=True, activation='tanh'))
        model.add(Dropout(0.2, training = True))
        model.add(GRU(
                    units=50,
                    return_sequences=False, activation='tanh'))
        model.add(Dropout(0.2, training = True))
        model.add(Dense(units=1, activation='relu'))
        model.compile(loss='mean_squared_error', optimizer= RMSprop(lr = 0.01), metrics=['mse'])
        model.fit(X_train, y_train, epochs=num_epochs, batch_size=200, validation_split=0.2, verbose=2, callbacks=[tensorboard, tensorboard2])
        #EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
        model.save("regression_model_saved")
    else:
        model = load_model("regression_model_saved")
    
    model.summary()
    scores = model.evaluate(X_train, y_train, verbose=1, batch_size=200)
    print('Accumulated loss: {}'.format(scores[1]))
    y_pred = model.predict(X_test)
    #print('Accuracy of model on test data: ',accuracy_score(y_test,y_pred))
    #print('Confusion Matrix: \n',confusion_matrix(y_test,y_pred))
    predict_RULs(numOfUnits, model)


RUL_regression(num_epochs=300, numOfUnits = 100, load_ex_model = 0)

#binary_classification(1, 0)


'''
xEpochs = np.arange(numEpochs)
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Training loss and validation loss')
ax1.plot(xEpochs, h.history['loss'])
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Training loss')
#ax2.set_xlabel('Epochs')
ax2.set_ylabel('Validation loss')
ax2.plot(xEpochs, h.history['val_loss'])
plt.show()
'''