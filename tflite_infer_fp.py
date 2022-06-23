import os
import glob
import cv2
import csv
import numpy as np
import tensorflow as tf
import argparse
import tqdm


def norm(img):
    std, mean = np.array(0.5), np.array(0.5)
    img = np.asarray(img).astype(np.float32)/255.
    img = (img - mean) / std
    return img


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, default='.tflite/model_float16_quant.tflite')
    argparser.add_argument("--data_path", type=str, default='./test_data/')
    args = argparser.parse_args()
    
    inter = tf.lite.Interpreter(args.model)
    inter.allocate_tensors()
    input_details = inter.get_input_details()
    output_details = inter.get_output_details()
    
    img_paths = glob.glob(os.path.join(args.data_path, '*.jpg'))

    data = list()
    with open(os.path.join(args.data_path, 'val.csv')) as f:
        reader = csv.reader(f)
        for line in reader:
            data.append(line)
    label2num = {}
    with open(os.path.join(args.data_path, 'label.txt'), 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            self.label2num[line.strip()] = i

    paths = list(map(lambda x: x[0], data)) 
    labels = list(map(lambda x: x[1], data))
    labels = list(map(lambda x: label2num[x], labels))
    
    imgs = list()
    _, w, h, _ = input_details[0]['shape']
    for path in tqdm.tqdm(paths):
        img = cv2.imread('./fashion-dataset/images/' + path.split('/')[-1])
        img = cv2.resize(img, (w, h))
        img = norm(img)
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)
        imgs += [img]
    print("Data ready!!")
    
    num_correct= 0
    for i in tqdm.tqdm(range(len(paths))):
        inter.set_tensor(input_details[0]['index'], imgs[i])
        inter.invoke()
        output_data = inter.get_tensor(output_details[0]['index'])
        pred = np.argmax(output_data)
        num_correct += (pred == labels[i])
    
    print('Accuracy: {:.3f}'.format(num_correct / len(paths) * 100))

if __name__ == '__main__':
    main()
