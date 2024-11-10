import os
import json
import cv2
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from foolbox import TensorFlowModel, Model, attacks
import numpy as np
from PIL import Image
import eagerpy as ep
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

SAVE_PATH = ''
VAL_IMG_PATH = ''


os.makedirs(SAVE_PATH, exist_ok=True)

model = ResNet50(weights='imagenet')

def get_label():
    file_path = 'ILSVRC2012_validation_ground_truth.txt'
    # Open the file for reading
    label_list = [-1]
    with open(file_path, 'r') as file:
        file_contents = file.readlines()
    for label in file_contents:
        label_list.append(int(label[:-1]))

    return label_list

def apply_single_attack(image, label, epsilons, attack):
    model = tf.keras.applications.ResNet50(weights="imagenet")
    pre = dict(flip_axis=-1, mean=[104.0, 116.0, 123.0])  
    fmodel: Model = TensorFlowModel(model, bounds=(0, 255), preprocessing=pre)
    fmodel = fmodel.transform_bounds((0, 1))

    image = tf.reshape(image, shape=(1, 224, 224, 3))
    label_np = label.numpy()
    label_tensor = tf.constant(label_np, dtype=tf.int64)
    label_ep = ep.astensor(label_tensor)


    _ , clipped, is_adv = attack(fmodel, image, label_ep, epsilons=epsilons)
    return clipped, is_adv, epsilons

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return x

def get_predicted_label(label_name):
    with open('mappings.json', 'r') as f:
        data = json.load(f)

    for key, value in data.items():
        if label_name in value['id']:
            return int(key)


if __name__ == '__main__':
    
    for img_path in os.listdir(VAL_IMG_PATH):

        print(img_path)
        image_number = int(img_path.split('_')[2][:-5])
        img_path = os.path.join(VAL_IMG_PATH, img_path)
        print(f"Iteration: {itr}")


        image = Image.open(img_path)
        image = image.resize((224,224))
        image = image.convert('RGB')
        image_np = np.array(image) / 255.0
        image_tensor = tf.convert_to_tensor(image_np, dtype=tf.float32)
        epsilons = [
           0.01,
            0.1,
           0.3,
          
        ]
        attack_dict = {
            "fgsm": attacks.LinfFastGradientAttack(),
             "bim" : attacks.LinfBasicIterativeAttack(),
             "pgd" : attacks.LinfProjectedGradientDescentAttack(),

            }

        label_list = get_label()
     
        print(f'img -> {img_path} | Original Label -> {label_list[image_number]}')
        for key, attack in attack_dict.items():
            os.makedirs(os.path.join(SAVE_PATH, key), exist_ok=True)
            clipped, is_adv, epsilons = apply_single_attack(image_tensor, tf.constant([label_list[image_number]]), epsilons, attack)

            for i,result in enumerate(is_adv):
                if result.numpy()[0] == True:
                    image_np = clipped[i][0].numpy()
                    image_np = image_np*255
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f"{SAVE_PATH}/{key}/image_{image_number}_label_{label_list[image_number]}_ep_{epsilons[i]}_attack_{key}.jpg", image_np)

    


       

