import keras_tuner
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def Mobilenet_v2(input_size,weights,Dropout_rate,Trainable,alpha = 0.35):
    base_model = keras.applications.MobileNetV2(
        input_shape=(input_size, input_size, 3),
        alpha=alpha,
        weights=weights,
        include_top=False
    )
    inputs = keras.Input(shape=(input_size, input_size,3))

    scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
    x = scale_layer(inputs)
    x = base_model(x, training=False)
    if Trainable:
        base_model.trainable =True
    else:
        base_model.trainable = False
        print("特征层已冻结")
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(Dropout_rate, name='Dropout')(x)
    outputs = keras.layers.Dense(15, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    return model


class HyperMobilenet_v2(keras_tuner.HyperModel):
    def __init__(self,input_size,batch_size,train_root):
        super(keras_tuner.HyperModel, self).__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.train_root = train_root
    def model(self,Dropout_rate,lr):
        model = Mobilenet_v2(
            input_size=self.input_size,
            weights='imagenet',
            Dropout_rate=Dropout_rate,
            Trainable=True
        )
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                    loss=keras.losses.CategoricalCrossentropy(),
                    metrics=["accuracy"])
        return model
    def build(self, hp):
        lr = hp.Float("lr",min_value=1e-5,max_value=1e-3,sampling="log")
        Dropout_rate = hp.Choice("Dropout_rate",values =[0.1,0.2,0.3,0.4,0.5,0.6])
        return self.model(Dropout_rate,lr) 
    
    def dataset(self, hp):

        train_root = self.train_root

        zoom_range = hp.Float("zoom_range",min_value=0.1,max_value=0.3,sampling="linear")
        channel_shift_range = hp.Int("channel_shift_range",min_value=0,max_value=30,step=10)

        train_generator = ImageDataGenerator(rotation_range=360,
                                            zoom_range =zoom_range,
                                            horizontal_flip = True,
                                            validation_split =0.2,
                                            channel_shift_range =channel_shift_range
                                            )
        train_dataset = train_generator.flow_from_directory(batch_size=self.batch_size,
                                                            directory=train_root,
                                                            shuffle=True,
                                                            target_size=(self.input_size,self.input_size),
                                                            subset='training')
        valid_dataset = train_generator.flow_from_directory(batch_size=self.batch_size,
                                                            directory=train_root,
                                                            shuffle=True,
                                                            target_size=(self.input_size,self.input_size),
                                                            subset='validation')
        return train_dataset,valid_dataset

    def fit(self, hp, model,**kwargs):
        
        train_dataset,valid_dataset = self.dataset(hp)

        return model.fit(
            train_dataset,
            validation_data=valid_dataset,
            **kwargs
        ) 

tuner = keras_tuner.BayesianOptimization(
    hypermodel =HyperMobilenet_v2(input_size=128,batch_size=128,train_root='./train/'),
    objective="val_accuracy",
    max_trials=50,
    overwrite=True,
    directory="my_dir",
    project_name="tune_hypermodel",
)
tuner.search(epochs=5,workers=8)

hypermodel = HyperMobilenet_v2(input_size=128,batch_size=128,train_root='./train/')
best_hp = tuner.get_best_hyperparameters()[0]
model = hypermodel.build(best_hp)
save_path = './models_save/%s' % (time.strftime('%Y_%m_%d_%H_%M_%S'))
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=10,verbose=1)
early_stop =keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10,verbose=1)
save_weights = keras.callbacks.ModelCheckpoint(save_path + "/model_{epoch:02d}_{val_accuracy:.4f}.h5",
                                                   save_best_only=True, monitor='val_accuracy')

hypermodel.fit(
    best_hp, 
    model,
    epochs=100,
    workers =8,
    callbacks=[save_weights,early_stop,reduce_lr]
    )