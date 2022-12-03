from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from IPython.display import display, HTML
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, Conv1D
from keras.utils import np_utils
from keras.layers.recurrent import LSTM, SimpleRNN
from sklearn.decomposition import PCA
from keras.optimizers import SGD
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau

#------------------------------------------------------------------
# 標準參數預設

pd.options.display.float_format = '{:.1f}'.format
sns.set()  # Default seaborn look and feel
plt.style.use('ggplot')
print('keras version ', keras.__version__)
# 設置LABEL參數

LABELS = ['leave', 'sit', 'lying']

# 設置WINDOW SIZE
TIME_PERIODS = 50
# 設置WINDOW SIZE 重疊部分
STEP_DISTANCE = 25

#------------------------------------------------------------------


#讀取CSV檔案設置
def read_data(file_path):

    column_names = ['user', 'time', 'strengh', 'plot', 'difference', 'active']
    df = pd.read_csv(file_path, header=None, names=column_names)

    # Last column has a ";" character which must be removed ...
    df['difference'].replace(regex=True,
                             inplace=True,
                             to_replace=r';',
                             value=r'')
    # ... and then this column must be transformed to float explicitly
    df['difference'] = df['difference'].apply(convert_to_float)

    # This is very important otherwise the model will not fit and loss
    # will show up as NAN
    df.dropna(axis=0, how='any', inplace=True)

    return df


def convert_to_float(x):

    try:
        return np.float(x)
    except:
        return np.nan


def show_basic_dataframe_info(dataframe):

    # Shape and how many rows and columns
    print('Number of columns in the dataframe: %i' % (dataframe.shape[1]))
    print('Number of rows in the dataframe: %i\n' % (dataframe.shape[0]))


# Load data set containing all the data from csv
df = read_data('C:/Users/sokel/Desktop/c/論文/demo/lstmtest.csv')

#------------------------------------------------------------------

# Show how many training examples exist for each of the six activities
df['active'].value_counts().plot(kind='bar',
                                 title='Training Examples by Activity Type')
plt.show()
# Better understand how the recordings are spread across the different
# users who participated in the study
df['user'].value_counts().plot(kind='bar', title='Training Examples by User')
plt.show()

#------------------------------------------------------------------


def plot_activity(activity, data):

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 10), sharex=True)
    plot_axis(ax0, data['time'], data['strengh'], 'X-Axis')
    plot_axis(ax1, data['time'], data['plot'], 'Y-Axis')
    plot_axis(ax2, data['time'], data['difference'], 'Z-Axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()


def plot_axis(ax, x, y, title):

    ax.plot(x, y, 'r')
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)


for activity in np.unique(df['active']):
    subset = df[df['active'] == activity][:180]
    plot_activity(activity, subset)

#------------------------------------------------------------------

# Define column name of the label vector
LABEL = 'ActivityEncoded'
# Transform the labels from String to Integer via LabelEncoder
le = preprocessing.LabelEncoder()
# Add a new column to the existing DataFrame with the encoded values
df[LABEL] = le.fit_transform(df['active'].values.ravel())

df_test = df[df['user'] >= 4]
df_train = df[df['user'] <= 4]

# Normalize features for training data set (values between 0 and 1)
# Surpress warning for next 3 operation
pd.options.mode.chained_assignment = None  # default='warn'
df_train['strengh'] = df_train['strengh'] / df_train['strengh'].max()
df_train['plot'] = df_train['plot'] / df_train['plot'].max()
df_train['difference'] = df_train['difference'] / df_train['difference'].max()
# Round numbers
df_train = df_train.round({'strengh': 4, 'plot': 4, 'difference': 4})


def create_segments_and_labels(df, time_steps, step, label_name):

    # x, y, z acceleration as features
    N_FEATURES = 3
    # Number of steps to advance in each iteration (for me, it should always
    # be equal to the time_steps in order to have no overlap between segments)
    # step = time_steps
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        xs = df['strengh'].values[i:i + time_steps]
        ys = df['plot'].values[i:i + time_steps]
        zs = df['difference'].values[i:i + time_steps]

        # Retrieve the most often used label in this segment
        label = stats.mode(df[label_name][i:i + time_steps])[0][0]
        segments.append([xs, ys, zs])
        labels.append(label)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(
        -1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels


x_train, y_train = create_segments_and_labels(df_train, TIME_PERIODS,
                                              STEP_DISTANCE, LABEL)

print('x_train shape: ', x_train.shape)
print(x_train.shape[0], 'training samples')
print('y_train shape: ', y_train.shape)

print('x_train shape: ', x_train.shape)
print(x_train.shape[0], 'training samples')
print('y_train shape: ', y_train.shape)

num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
num_classes = le.classes_.size
print(list(le.classes_))

input_shape = (num_time_periods * num_sensors)
x_train = x_train.reshape(x_train.shape[0], input_shape)
print('x_train shape:', x_train.shape)
print('input_shape:', input_shape)

x_train = x_train.astype('float32')
y_train = y_train.astype('float32')

y_train_hot = np_utils.to_categorical(y_train, num_classes)
print('New y_train shape: ', y_train_hot.shape)

#------------------------------------------------------------------
#MODEL層設置

model_m = Sequential()
# Remark: since coreml cannot accept vector shapes of complex shape like
# [80,3] this workaround is used in order to reshape the vector internally
# prior feeding it into the network
model_m.add(Reshape((TIME_PERIODS, 3), input_shape=(input_shape, )))
model_m.add(Dropout(0.3))
model_m.add(LSTM(32, return_sequences=True))
model_m.add(Dropout(0.3))
model_m.add(Dense(10, activation='relu'))
model_m.add(Dropout(0.3))
model_m.add(Flatten())
model_m.add(Dense(num_classes, activation='softmax'))
keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
print(model_m.summary())

callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss',
        save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='accuracy', patience=1)
]

model_m.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

#------------------------------------------------------------------
#訓練參數設置

# Hyper-parameters
BATCH_SIZE = 600

EPOCHS = 100

# Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
history = model_m.fit(
    x_train,
    y_train_hot,
    batch_size=10,
    epochs=EPOCHS,
    callbacks=[],
    validation_split=0.2,
)

history2 = model_m.fit(
    x_train,
    y_train_hot,
    batch_size=20,
    epochs=EPOCHS,
    callbacks=[],
    validation_split=0.2,
)

history3 = model_m.fit(
    x_train,
    y_train_hot,
    batch_size=30,
    epochs=EPOCHS,
    callbacks=[],
    validation_split=0.2,
)

history4 = model_m.fit(
    x_train,
    y_train_hot,
    batch_size=50,
    epochs=EPOCHS,
    callbacks=[],
    validation_split=0.2,
)

#------------------------------------------------------------------
#顯示訓練結果
plt.figure(figsize=(18, 8))

plt.plot(history.history['val_loss'], 'r', label='batch_size=10')
plt.plot(history2.history['val_loss'], 'y', label='batch_size=20')
plt.plot(history3.history['val_loss'], 'b', label='batch_size=30')
plt.plot(history4.history['val_loss'], 'g', label='batch_size=50')

plt.title('Model Loss ')
plt.ylabel('Loss ')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()

plt.axis([0, 100, 0.0, 0.9])
plt.show()

plt.figure(figsize=(18, 8))

plt.plot(history.history['loss'], 'r--', label='Loss of training data')
plt.plot(history.history['val_loss'], 'b--', label='Loss of validation data')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()

plt.show()

#------------------------------------------------------------------

# Print confusion matrix for training data
y_pred_train = model_m.predict(x_train)
# Take the class with the highest probability from the train predictions
max_y_pred_train = np.argmax(y_pred_train, axis=1)
print(classification_report(y_train, max_y_pred_train))

df_test['strengh'] = df_test['strengh'] / df_test['strengh'].max()
df_test['plot'] = df_test['plot'] / df_test['plot'].max()
df_test['difference'] = df_test['difference'] / df_test['difference'].max()

df_test = df_test.round({'strengh': 4, 'plot': 4, 'difference': 4})

x_test, y_test = create_segments_and_labels(df_test, TIME_PERIODS,
                                            STEP_DISTANCE, LABEL)

# Set input_shape / reshape for Keras
x_test = x_test.reshape(x_test.shape[0], input_shape)

x_test = x_test.astype('float32')
y_test = y_test.astype('float32')

y_test = np_utils.to_categorical(y_test, num_classes)

score = model_m.evaluate(x_test, y_test, verbose=1)

print('\nAccuracy on test data: %0.2f' % score[1])
print('\nLoss on test data: %0.2f' % score[0])


#------------------------------------------------------------------
def show_confusion_matrix(validations, predictions):

    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        matrix,
        cmap='coolwarm',
        linecolor='white',
        linewidths=1,
        xticklabels=LABELS,
        yticklabels=LABELS,
        annot=True,
        fmt='d',
    )
    plt.title('LSTM Confusion Matrix')
    plt.ylabel('Real Label')
    plt.xlabel('Detected Label')
    plt.show()


y_pred_test = model_m.predict(x_test)
# Take the class with the highest probability from the test predictions
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(y_test, axis=1)

show_confusion_matrix(max_y_test, max_y_pred_test)

print(classification_report(max_y_test, max_y_pred_test))

prediction = model_m.predict_classes(x_test)
print(prediction)

print('\nPrediction from Keras:')
test_record = x_test[1].reshape(1, input_shape)
keras_prediction = np.argmax(model_m.predict(test_record), axis=1)
print(model_m.predict(test_record))
print(le.inverse_transform(keras_prediction)[0])

#------------------------------------------------------------------
model_m.save('0722-1.h5')  #modelh存取
#------------------------------------------------------------------
