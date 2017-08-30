**Note**: The following materials are my arrangement about Keras-introduction from Yiming Lin's Youtube sharing: https://www.youtube.com/watch?v=OUMDUq5OJLg&t=172s. 
Only for learning purpose. **If there is infringement please contact me to delete**.
### Why Keras? 
 **Always remember using KEras & TEnsorflow (KETE) combo rocks.**
1. Perfect Integration with Tensorflow
2. High-level abstraction
3. Well-written document: https://keras.io

### Keras Working Pipeline

> 1. ***Model definition*** (0:15:00)
> `model  = Sequential() `
> `model.add()`
> 2. ***Model compilation*** (0:15:15)
>*by default*
> `model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])`
> *by self-define*
> `from keras.optimizers import SGD`
>`model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.01,momentum=0.9,nesterov=True))`
> 3. ***Training*** 
> `model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)`
> 4. ***Prediction and Evaluation***
> *Evaluate your performance in one line:*
> `loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)`
> *Or generate predictions on new data*
> `classes = model.predict_classes(X_test, batch_size = 32)`
> `proba = model.predict_proba(X_test, batch_size = 32)`


### Keras Utilities 
#### Preprocessing 

*Keras Preprocessing provides useful data augmentation methods for Sequence, Text and Image data. Take image for example, some augmentation are normally done:*

 - Flipping
 - Shearing
 - Rotation
 - Rescaling to [0,1]
 - Etc.

**keras.preprocessing.image,imageDataGenerator**

```python
train_datagen =  ImageDataGenerator(
rescale=1./255,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)

train_generator = train_daagen.flow_from_directory(
'data/train',
target_size=(150,150),
batch_size=32,
class_mode='binary') 
#'binary' means that: data/train/dogs---class_0, data/train/cats---class_1

validation_generator = test_datagen.flow_from_directory(
'data/validation',
target_size=(150,150),
batch_size=32,
class_mode='binary'
)

model.fit_generator(
train_generator,
sample_per_epoch=2000,
nb_epoch=50,
validation_data = validation_generator,
nb_val_samples=800
)
```

#### Application

> Keras Applications are **deep learning models** that are made available alongside **pre-trained weights**. These models can be used for prediction, feature extraction, and fine-tuning.Weights are downloaded automatically when instantiating a model. **They are stored at ~/.keras/models/**.

```python
# Extract features with VGG16
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

model = VGG16(weights = 'imagenet', include_top=False)
# Keras will download the VGG16 weights when your specipy VGG16
# include_top = False means you use it for extracting features for all Convs
# weights path = '.keras/models/weights.h5'
img_path = 'elephant.jpg'
img = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict()

```

### Keras Example
#### Cats and Dogs Classification in Jupyter Notebook
[Keras 2.0 release notes](https://github.com/fchollet/keras/wiki/Keras-2.0-release-notes)
[Keras-Learning_Notes](https://github.com/GuokaiLiu/Keras-Learning_Notes)