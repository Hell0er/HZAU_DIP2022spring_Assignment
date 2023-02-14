import os
from turtle import color
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.applications import VGG16

'''
激活函数的作用
optimizer: 调整参数，使得损失函数趋近全局最小
'''

train_data_dir = 'D:/NotOnlyCode/DIP/downloads/train'
val_data_dir = 'D:/NotOnlyCode/DIP/downloads/val'
batch_size = 32

def VGG_model():

    conv_base = VGG16(weights='imagenet', include_top = False, input_shape=(150,150,3))
    conv_base.trainable = False

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',  # 定义模型的loss func，optimizer，
                  optimizer=optimizers.Adam(),  # 使用默认的lr=0.001
                  metrics=['acc'])  # 主要优化accuracy

    model.summary()
    return model
    
def load_image():
    train_datagen = ImageDataGenerator(
        rescale=1./225,
        shear_range=0.2,
        zoom_range=0.2,     #随机缩放扩大
        rotation_range=30,
        horizontal_flip=True,
        vertical_flip=True,)

    #验证集的数据不做图像增强处理
    val_datagen = ImageDataGenerator(
        rescale=1./225)
    
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (150, 150),
        batch_size=batch_size,
        class_mode='categorical' #多分类的评估指标
        )
    val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size = (150, 150),
        batch_size=batch_size,
        class_mode='categorical'
        )

    return train_generator, val_generator

#绘图
def plot_image(history):

    #No processing
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    
    plt.plot(epochs, loss, 'bo',label = 'Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss', color='#FF0000')
    plt.title('train and validation loss')
    plt.legend()
    plt.savefig('./plot_images/loss.jpg')
    plt.close()

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc' ,color='#FF0000')
    plt.legend()
    plt.savefig('./plot_images/acc.jpg')
    plt.close()

    #Smooth processing
    def smooth_curve(points, factor=0.8):
        smoothed_points = []
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(previous * factor + point * (1 - factor))
            else:
                smoothed_points.append(point)
        return smoothed_points

    plt.plot(epochs, smooth_curve(loss), 'bo',label = 'Training loss')
    plt.plot(epochs, smooth_curve(val_loss), 'b', label='Validation loss', color='#FF0000')
    plt.title('train and validation loss')
    plt.legend()
    plt.savefig('./plot_images/sommth_loss.jpg')
    plt.close()

    plt.plot(epochs, smooth_curve(acc), 'bo', label='Training acc')
    plt.plot(epochs, smooth_curve(val_acc), 'b', label='Validation acc' ,color='#FF0000')
    plt.legend()
    plt.savefig('./plot_images/smooth_acc.jpg')
    plt.close()


if __name__ == "__main__":
    model = VGG_model()
    
    train_generator, val_generator = load_image()
    for data_batch, lable_batch in train_generator:
        print('data_batch_shape:', data_batch.shape)
        print('label_batch_shape:', lable_batch.shape)
        break
    # data_batch_shape: (15, 150, 150, 3)
    # label_batch_shape: (15, 3)


    print(">>>>>>>>>>>开卷>>>>>>>>>>>>>")
    # 拟合模型

    '''
    steps_pre_epoch为默认值是才可以对shuffle参数修改
    '''
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=50,
        epochs=80,
        validation_data=val_generator,
        validation_steps=32,
        # shuffle=True
    )   

    plot_image(history)




    


