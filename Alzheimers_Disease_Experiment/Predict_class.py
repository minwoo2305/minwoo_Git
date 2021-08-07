from keras.models import load_model
import os
from keras.preprocessing import image
import numpy as np

num = '11'
acc = '62.5'

path = 'E:/MCI_vs_AD/' + num + '/'
model_name = 'save_model_' + acc + '.h5'

model = load_model(path + '/' + model_name)

data_path = 'C:/Users/minwo/바탕 화면/gc_data/img_data/' + num

ad_img_dir = data_path + '/' + 'AD'
mci_img_dir = data_path + '/' + 'MCI'
sci_img_dir = data_path + '/' + 'SCI'

ad_img = []
for img_name in os.listdir(ad_img_dir):
    img_path = os.path.join(ad_img_dir, img_name)
    img = image.load_img(img_path, target_size=(100, 100))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    ad_img.append(img_tensor)

mci_img = []
for img_name in os.listdir(mci_img_dir):
    img_path = os.path.join(mci_img_dir, img_name)
    img = image.load_img(img_path, target_size=(100, 100))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    mci_img.append(img_tensor)

sci_img = []
for img_name in os.listdir(sci_img_dir):
    img_path = os.path.join(sci_img_dir, img_name)
    img = image.load_img(img_path, target_size=(100, 100))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    sci_img.append(img_tensor)

ad_img_result = []
mci_img_result = []
sci_img_result = []

for img in ad_img:
    ad_img_result.append(model.predict_classes(img)[0])

for img in mci_img:
    mci_img_result.append(model.predict_classes(img)[0])

for img in sci_img:
    sci_img_result.append(model.predict_classes(img)[0])

print(ad_img_result)
print(mci_img_result)
# print(sci_img_result)