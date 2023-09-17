from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow.keras.backend as kbackend
from numpy import loadtxt


def create_default_model(input_shape):
    ### make it general
    model = Sequential(name='queueTime')
    model.add(Input(shape=(input_shape,)))
    model.add(Dense(30, activation='relu', name='Hidden1'))
    model.add(Dense(100, activation='relu', name='Hidden2'))
    model.add(Dense(100, activation='relu', name='Hidden3'))
    model.add(Dense(1, activation='linear', name='Output'))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def create_checkpoint_cb(h5_file):
    checkpoint_cb = ModelCheckpoint(h5_file, save_best_only=True)
    early_stopping_cb = EarlyStopping(patience=30, restore_best_weights=True)
    return checkpoint_cb,early_stopping_cb

def train_model_cb_cp(X_historydata_norm, Y_waittimedata_train, model,checkpoint_cb,
                early_stopping_cb, h5_file, epochs, batch_size,validation_spilt):

    hist = model.fit(X_historydata_norm, Y_waittimedata_train, batch_size=batch_size, epochs=epochs,
                    validation_split=validation_spilt, callbacks=[checkpoint_cb,early_stopping_cb])
    model = load_model(h5_file)
    print(model.summary())
    return model, hist

def train_model(X_historydata_norm, Y_waittimedata_train, model, epochs,batch_size,validation_spilt):
    hist = model.fit(X_historydata_norm, Y_waittimedata_train, batch_size=batch_size, epochs=epochs,
                    validation_split=validation_spilt)
    #model = load_model(h5_file)
    print(model.summary())
    return model, hist

def create_model_architecture(model):
    model.add(Dense(30, activation='relu', name='Hidden1'))
    model.add(Dense(100, activation='relu', name='Hidden2'))
    model.add(Dense(100, activation='relu', name='Hidden3'))
    model.add(Dense(1, activation='linear', name='Output'))
    return model

def custom_asymmetric_loss_2(y_true, y_pred):
    y_true = kbackend.cast(y_true, kbackend.floatx())
    diff = y_pred - y_true

    greater = kbackend.greater(diff, 0)
    greater = kbackend.cast(greater, kbackend.floatx())  # 0 for lower, 1 for greater
    greater = greater + 1  # 1 for lower, 2 for greater

    # use some kind of loss here, such as mse or mae, or pick one from keras
    # using mse:
    return kbackend.mean(greater * kbackend.square(diff))

def custom_asymmetric_loss_3(y_true, y_pred):
    n=3
    y_true = kbackend.cast(y_true, kbackend.floatx())
    diff = y_pred - y_true
    print(diff)
    diff1 = kbackend.switch(diff>0,n*kbackend.square(diff), kbackend.square(diff))
    print(diff1)
    return kbackend.mean(kbackend.cast(diff1,kbackend.floatx()))

def custom_asymmetric_loss_4(y_true, y_pred):
    n=4
    y_true = kbackend.cast(y_true, kbackend.floatx())
    diff = y_pred - y_true
    print(diff)
    diff1 = kbackend.switch(diff>0,n*kbackend.square(diff), kbackend.square(diff))
    print(diff1)
    return kbackend.mean(kbackend.cast(diff1,kbackend.floatx()))

def custom_asymmetric_loss_5(y_true, y_pred):
    n=5
    y_true = kbackend.cast(y_true, kbackend.floatx())
    diff = y_pred - y_true
    print(diff)
    diff1 = kbackend.switch(diff>0,n*kbackend.square(diff), kbackend.square(diff))
    print(diff1)
    return kbackend.mean(kbackend.cast(diff1,kbackend.floatx()))

def create_model_with_asymmetric_loss_2(input_shape):
    ### make it general
    model = Sequential(name='queueTime')
    model.add(Input(shape=(input_shape,)))
    model = create_model_architecture(model)
    model.compile(optimizer='rmsprop', loss=custom_asymmetric_loss_2, metrics=['mae'])
    return model

def create_model_with_asymmetric_loss_3(input_shape):
    ### make it general
    model = Sequential(name='queueTime')
    model.add(Input(shape=(input_shape,)))
    model = create_model_architecture(model)
    model.compile(optimizer='rmsprop', loss=custom_asymmetric_loss_3, metrics=['mae'])
    return model

def create_model_with_asymmetric_loss_4(input_shape):
    ### make it general
    model = Sequential(name='queueTime')
    model.add(Input(shape=(input_shape,)))
    model = create_model_architecture(model)
    model.compile(optimizer='rmsprop', loss=custom_asymmetric_loss_4, metrics=['mae'])
    return model
def create_model_with_asymmetric_loss_5(input_shape):
    ### make it general
    model = Sequential(name='queueTime')
    model.add(Input(shape=(input_shape,)))
    model = create_model_architecture(model)
    model.compile(optimizer='rmsprop', loss=custom_asymmetric_loss_5, metrics=['mae'])
    return model

def create_new_arch(model):
    model.add(Dense(32, activation='relu', name='Hidden1'))
    model.add(Dense(100, activation='relu', name='Hidden2'))
    model.add(Dense(512, activation='relu', name='Hidden3'))
    model.add(Dense(100, activation='relu', name='Hidden4'))
    model.add(Dense(1, activation='linear', name='Output'))
    return model


def create_new_model(input_shape):
    ### make it general
    model = Sequential(name='queueTime')
    model.add(Input(shape=(input_shape,)))
    model = create_new_arch(model)
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def create_new_arch_param(model,param):

    model.add(Dense(32, activation='relu', name='Hidden1'))
    for i in range(param):
        model.add(Dense(100, activation='relu', name='Hidden'+str(i+1)))
    model.add(Dense(512, activation='relu', name='Hidden'+str(param+1)))
    model.add(Dense(100, activation='relu', name='Hidden'+str(param+1)))
    model.add(Dense(1, activation='linear', name='Output'))
    return model

def create_new_model_param(input_shape,param):
    ### make it general
    model = Sequential(name='queueTime')
    model.add(Input(shape=(input_shape,)))
    model = create_new_arch_param(model,param)
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model