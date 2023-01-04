from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import time
import os
import pandas as pd

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

epochs = 5
input_size=128
lr =0.0001
Dropout_rate=0.3
batch_size =128



train_root = "./train/"

train_generator = ImageDataGenerator(rotation_range=360,
                                     zoom_range  =0.2,
                                     horizontal_flip = True,
                                     validation_split =0.2
                                     )
train_dataset = train_generator.flow_from_directory(batch_size=batch_size,
                                                    directory=train_root,
                                                    shuffle=True,
                                                    target_size=(input_size,input_size),
                                                    subset='training')
valid_dataset = train_generator.flow_from_directory(batch_size=batch_size,
                                                   directory=train_root,
                                                    shuffle=True,
                                                    target_size=(input_size,input_size),
                                                    subset='validation')
print(train_dataset.class_indices)

model = Mobilenet_v2(input_size,weights='imagenet',Dropout_rate=Dropout_rate,Trainable=True)
model.summary()

model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
              loss=keras.losses.CategoricalCrossentropy(),
              metrics=["accuracy"])

save_path = './models_save/%s' % (time.strftime('%Y_%m_%d_%H_%M_%S'))

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=10,verbose=1)
early_stop =keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10,verbose=1)
save_weights = keras.callbacks.ModelCheckpoint(save_path + "/model_{epoch:02d}_{val_accuracy:.4f}.h5",
                                                   save_best_only=True, monitor='val_accuracy')
hist = model.fit(train_dataset, 
                 epochs=epochs,
                 validation_data=valid_dataset,
                  callbacks=[save_weights,early_stop,reduce_lr]
                 )

plt.figure()
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(f'{save_path}/acc')

# 绘制训练 & 验证的损失值
plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(f'{save_path}/loss')

df = pd.DataFrame.from_dict(hist.history)
df.to_csv(f'{save_path}/hist.csv', encoding='utf-8', index=False)
