import os
import cv2
import numpy as np
import tensorflow as tf
from foolbox import TensorFlowModel, attacks
import eagerpy as ep
import tensorflow_hub as hub


NUM_CLASSES = 10  
# Load pre-trained BiT model (BiT-M-R101x3) 
model_url = "https://tfhub.dev/google/bit/m-r101x3/1"
module = hub.KerasLayer(model_url)
# Define optimizer and loss
SCHEDULE_BOUNDARIES = [3000, 6000, 9000]
lr = 0.001
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=SCHEDULE_BOUNDARIES, 
    values=[lr, lr*0.1, lr*0.001, lr*0.0001]
)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Create the model
class MyBiTModel(tf.keras.Model):
    def __init__(self, num_classes, module):
        super().__init__()
        self.num_classes = num_classes
        self.head = tf.keras.layers.Dense(num_classes, kernel_initializer='zeros')
        self.bit_model = module
  
    def call(self, images):
        bit_embedding = self.bit_model(images)
        return self.head(bit_embedding)




SAVE_PATH = ''
VAL_IMG_PATH = ''



os.makedirs(SAVE_PATH, exist_ok=True)

def get_label():
    file_path = 'cifar10_dataset/ground_truth.txt'
    label_list = []
    with open(file_path, 'r') as file:
        for line in file:
            image_id, label = line.strip().split()
            label_list.append(int(label))
    return label_list




def apply_single_attack(image, label, epsilons, attack, model):
    pre = dict(flip_axis=-1, mean=[125.306918046875, 122.950394140625, 113.86538318359375])  # CIFAR-10 mean values
    fmodel = TensorFlowModel(model, bounds=(0, 255), preprocessing=pre)
    fmodel = fmodel.transform_bounds((0, 1))

    image = tf.image.resize(image, (128, 128))  # Resize CIFAR-10 images to 128x128 (adjust as per your model input size)
    image = tf.reshape(image, shape=(1, 128, 128, 3))  # Reshape to match model input shape

    label_np = label.numpy()
    label_tensor = tf.constant(label_np, dtype=tf.int64)
    label_ep = ep.astensor(label_tensor)

    _, clipped, is_adv = attack(fmodel, image, label_ep, epsilons=epsilons)
    return clipped, is_adv, epsilons

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))  # Resize images to match model input size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return img

if __name__ == '__main__':
    # Load BiT model here
    model = MyBiTModel(num_classes=NUM_CLASSES, module=module)
    model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=['accuracy'] )

    dummy_input = tf.zeros((1, 128, 128, 3))  # Example input shape
    _ = model(dummy_input)

    model.load_weights("")  # Load the trained model weights

    for i, img_path in enumerate(os.listdir(VAL_IMG_PATH)):

        image_number = int(img_path.split('.')[0]) 

        print(f"Iteration: {i}")
        img_path = os.path.join(VAL_IMG_PATH, img_path)

        image_np = preprocess_image(img_path)
        image_tensor = tf.convert_to_tensor(image_np, dtype=tf.float32)
        epsilons = [0.01, 0.1, 0.3]

        attack_dict = {
            "fgsm": attacks.LinfFastGradientAttack(),
             "bim" : attacks.LinfBasicIterativeAttack(),
            "pgd" : attacks.LinfProjectedGradientDescentAttack(),
        }

        label_list = get_label()

        for key, attack in attack_dict.items():
            os.makedirs(os.path.join(SAVE_PATH, key), exist_ok=True)
            clipped, is_adv, epsilons = apply_single_attack(image_tensor, tf.constant([label_list[image_number]]), epsilons, attack, model)
            for j, result in enumerate(is_adv):
                if result.numpy()[0] == True:
                    image_np = clipped[j][0].numpy() * 255
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                   # cv2.imwrite(f"{SAVE_PATH}/{key}/image_{i}_ep_{epsilons[j]}_attack_{key}.png", image_np)
                    cv2.imwrite(f"{SAVE_PATH}/{key}/image_{image_number}_label_{label_list[image_number]}_ep_{epsilons[j]}_attack_{key}.png", image_np)
