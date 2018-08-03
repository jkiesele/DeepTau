
from DeepJetCore.training.training_base import training_base
import keras
from keras.models import Model
from keras.layers import Dense,Dropout, Flatten,Concatenate, Convolution1D, LSTM, Multiply
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate


def my_model(Inputs,nclasses,nregressions,dropoutRate=0.1, batchmomentum=0.6, l2norm=1e-4):
    
    globalvars = BatchNormalization(momentum=0.6,name='globals_input_batchnorm') (Inputs[0])
    cpf    =     BatchNormalization(momentum=0.6,name='cpf_input_batchnorm')     (Inputs[1])
    npf    =     BatchNormalization(momentum=0.6,name='npf_input_batchnorm')     (Inputs[2])
    pt_inputs = Inputs[3] # don't normalise
    
    cpf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv0',
               kernel_regularizer=keras.regularizers.l2(0.001*l2norm))(cpf)
    cpf = Dropout(0.1*dropoutRate,name='cpf_dropout0')(cpf)                                                   
    cpf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv1',
               kernel_regularizer=keras.regularizers.l2(0.01*l2norm))(cpf)
    cpf = Dropout(0.1*dropoutRate,name='cpf_dropout1')(cpf)                                                   
    cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv2',
               kernel_regularizer=keras.regularizers.l2(0.01*l2norm))(cpf)
    cpf = Dropout(0.1*dropoutRate,name='cpf_dropout2')(cpf)                                                   
    cpf  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='cpf_conv3',
               kernel_regularizer=keras.regularizers.l2(0.01*l2norm))(cpf)
    cpf = BatchNormalization(momentum=0.6,name='cpf_conv_batchnorm')(cpf)  
    cpf = Dropout(dropoutRate,name='cpf_dropout3')(cpf)            

    
    npf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv0',
               kernel_regularizer=keras.regularizers.l2(0.001*l2norm))(npf)
    npf = Dropout(0.1*dropoutRate,name='npf_dropout0')(npf)                                                   
    npf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv1',
               kernel_regularizer=keras.regularizers.l2(0.01*l2norm))(npf)
    npf = Dropout(0.1*dropoutRate,name='npf_dropout1')(npf)                                                   
    npf  = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv2',
               kernel_regularizer=keras.regularizers.l2(0.01*l2norm))(npf)
    npf = Dropout(0.1*dropoutRate,name='npf_dropout2')(npf)                                                   
    npf  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='npf_conv3',
               kernel_regularizer=keras.regularizers.l2(0.01*l2norm))(npf)
    npf = BatchNormalization(momentum=0.6,name='npf_conv_batchnorm')(npf)
    npf = Dropout(dropoutRate,name='npf_dropout3')(npf) 
    
    cpf = LSTM(60,go_backwards=True,implementation=2, name='cpf_lstm',
               dropout=dropoutRate, recurrent_dropout=dropoutRate, activation='relu',
               #recurrent_activation='relu',
               return_sequences=True,
               kernel_regularizer=keras.regularizers.l2(l2norm))(cpf)
               
    cpf = LSTM(60,go_backwards=False,implementation=2, name='cpf_lstm2',
               dropout=dropoutRate, recurrent_dropout=dropoutRate, activation='relu',
               #recurrent_activation='relu',
               return_sequences=False,
               kernel_regularizer=keras.regularizers.l2(l2norm))(cpf)
               
    npf = LSTM(60,go_backwards=True,implementation=2, name='npf_lstm',
               dropout=dropoutRate, recurrent_dropout=dropoutRate, activation='relu',
               #recurrent_activation='relu',
               return_sequences=True,
               kernel_regularizer=keras.regularizers.l2(l2norm))(npf)
               
    npf = LSTM(60,go_backwards=True,implementation=2, name='npf_lstm2',
               dropout=dropoutRate, recurrent_dropout=dropoutRate, activation='relu',
               #recurrent_activation='relu',
               return_sequences=False,
               kernel_regularizer=keras.regularizers.l2(l2norm))(npf)
               
    #cpf = Flatten()(cpf)
    #npf = Flatten()(npf)
    
    x = Concatenate()([cpf,npf,globalvars])
    x = Dropout(dropoutRate,name='dropout_first')(x) 
    
    #x = Dense(200, activation='relu',kernel_initializer='lecun_uniform', name='dense0',
    #           kernel_regularizer=keras.regularizers.l2(l2norm))(x)
    #x = BatchNormalization(momentum=batchmomentum,name='dense_batchnorm0')(x)
    #x = Dropout(dropoutRate,name='dense_dropout0')(x)
    x = Dense(128, activation='relu',kernel_initializer='lecun_uniform', name='dense1',
               kernel_regularizer=keras.regularizers.l2(l2norm))(x)
    x = BatchNormalization(momentum=batchmomentum,name='dense_batchnorm1')(x)
    x = Dropout(dropoutRate,name='dense_dropout1')(x)
    #x = Dense(128, activation='relu',kernel_initializer='lecun_uniform', name='dense2',
    #           kernel_regularizer=keras.regularizers.l2(l2norm))(x)
    #x = BatchNormalization(momentum=batchmomentum,name='dense_batchnorm2')(x)
    #x = Dropout(dropoutRate,name='dense_dropout2')(x)
    x = Dense(64, activation='relu',kernel_initializer='lecun_uniform', name='dense3',
               kernel_regularizer=keras.regularizers.l2(l2norm))(x)
    x = BatchNormalization(momentum=batchmomentum,name='dense_batchnorm3')(x)
    x = Dropout(dropoutRate,name='dense_dropout3')(x)
    #x = Dense(64, activation='relu',kernel_initializer='lecun_uniform', name='dense4',
    #           kernel_regularizer=keras.regularizers.l2(l2norm))(x)
    #x = BatchNormalization(momentum=batchmomentum,name='dense_batchnorm4')(x)
    #x = Dropout(dropoutRate,name='dense_dropout4')(x)
    x = Dense(32, activation='relu',kernel_initializer='lecun_uniform', name='dense5',
               kernel_regularizer=keras.regularizers.l2(l2norm))(x)
    x = BatchNormalization(momentum=batchmomentum,name='dense_batchnorm5')(x)
    x = Dropout(dropoutRate,name='dense_dropout5')(x)
    
    ID_pred = Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    pt = Concatenate()([x,pt_inputs,globalvars]) #pt etc
    pt = Dense(32, activation='relu',
                kernel_initializer='lecun_uniform',
                name='dense_pt_1r',
               #kernel_regularizer=keras.regularizers.l2(l2norm)
               )(pt)
               
    pt = Dense(int(pt_inputs.shape[1]), activation='relu',
               kernel_initializer='lecun_uniform',
                name='dense_pt_3',
               #kernel_regularizer=keras.regularizers.l2(l2norm)
               )(pt)
    pt = Dense(256, activation='sigmoid',
               kernel_initializer='lecun_uniform',
                name='dense_pt_3',
               #kernel_regularizer=keras.regularizers.l2(l2norm)
               )(pt)
    pt = Dense(int(pt_inputs.shape[1]), activation='softsign',
               kernel_initializer='lecun_uniform',
                name='dense_pt_corr',
               #kernel_regularizer=keras.regularizers.l2(l2norm)
               )(pt)
               
    pt = Multiply()([pt,pt_inputs])
    pt = Concatenate()([x,pt_inputs,globalvars]) 
    pt = Dense(32, activation='relu',
                kernel_initializer=keras.initializers.random_normal(1./32., 1e-3),
                name='dense_pt_end',
               #kernel_regularizer=keras.regularizers.l2(l2norm)
               )(pt)
    
    pt_pred = Dense(1, activation='linear',kernel_initializer=keras.initializers.random_normal(1./16., 1e-3),name='pt_pred')(pt)
    
    predictions = [ID_pred,pt_pred]
    return Model(inputs=Inputs, outputs=predictions)


train=training_base(testrun=False,resumeSilently=False,renewtokens=True)
from Losses import relative_rms
additional_plots=['ID_pred_acc','val_ID_pred_acc',
                  'pt_pred_loss','val_pt_pred_loss',
                  'pt_pred_relative_rms', 'val_pt_pred_relative_rms']

metrics=['accuracy',relative_rms]


if not train.modelSet(): # allows to resume a stopped/killed training. Only sets the model if it cannot be loaded from previous snapshot

    train.setModel(my_model,dropoutRate=0.2, l2norm=1e-8)
    
    
    train.compileModel(learningrate=0.005,
                   loss=['categorical_crossentropy','mean_squared_error'],
                   loss_weights=[1, 1e-6],
                   metrics=metrics) 
            
  
print(train.keras_model.summary())  

class_weight=train.train_data.getClassWeights()
print(class_weight)
 
#exit()
model,history = train.trainModel(nepochs=2, 
                                 batchsize=1000,
                                 checkperiod=1, # saves a checkpoint model every N epochs
                                 verbose=1,
                                 additional_plots=additional_plots,
                                 class_weight=class_weight)

            
            
train.compileModel(learningrate=0.002,
                   loss=['categorical_crossentropy','mean_squared_error'],
                   loss_weights=[1, 1e-4],
                   metrics=metrics) 

print(train.keras_model.summary()) 
model,history = train.trainModel(nepochs=4, 
                                 batchsize=10000,
                                 checkperiod=5, # saves a checkpoint model every N epochs
                                 verbose=1,
                                 additional_plots=additional_plots,
                                 class_weight=class_weight)
    
    
    
train.compileModel(learningrate=0.0001,
                   loss=['categorical_crossentropy','mean_squared_error'],
                   loss_weights=[1, 1e-5],
                   metrics=metrics) 

model,history = train.trainModel(nepochs=10, 
                                 batchsize=10000,
                                 checkperiod=5, # saves a checkpoint model every N epochs
                                 verbose=1,
                                 additional_plots=additional_plots,
                                 class_weight=class_weight)

    
train.compileModel(learningrate=0.00003,
                   loss=['categorical_crossentropy','mean_squared_error'],
                   loss_weights=[1, 1e-5],
                   metrics=metrics) 

model,history = train.trainModel(nepochs=50, 
                                 batchsize=10000,
                                 checkperiod=5, # saves a checkpoint model every N epochs
                                 verbose=1,
                                 additional_plots=additional_plots,
                                 class_weight=class_weight)
    

