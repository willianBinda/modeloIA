import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.models import load_model

img_height, img_width = 384, 512
batch_size = 32

# def predict_image(model,img_path,img_height,img_width):
#     img = load_img(img_path,target_size=(img_height,img_width))
#     img_array = img_to_array(img)
#     img_array = np.expand_dims(img_array,axis=0)/255
#     prediction = model.predict(img_array)
#     return prediction

def predict_image(model,img_path,img_height,img_width):
    img = load_img(img_path,target_size=(img_height,img_width))

    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    return prediction

def get_class_label(prediction, class_indices):
    max_prob = np.max(prediction)
    class_label = None

    for label, index in class_indices.items():
        if prediction[0][index] == max_prob:
            class_label = label
            break
    return class_label

model = load_model('modeloIA.h5')

train_datagen = ImageDataGenerator(rescale=1.0 / 255)
# train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    './dataset/lixo',
    target_size=(img_height,img_width),
    batch_size=batch_size,
    class_mode='categorical')

class_indices = train_generator.class_indices

image_path = ''

prediction = predict_image(model,image_path,img_height,img_width)
class_label = get_class_label(prediction,class_indices)

if class_label is None:
    print("Não foi possível determinar a classe da imagem.")

print(f"A imagem '{image_path}' foi classificada como '{class_label}', com probabilidade {np.max(prediction)*100:.2f}%.")