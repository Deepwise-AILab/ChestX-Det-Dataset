import cv2
from torchvision import transforms
import pdb
import os
import torch
import numpy as np 
from collections import OrderedDict
from ptsemseg.pspnet import pspnet

Transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal 
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = "4"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.classes = ['Left Clavicle', 'Right Clavicle', 'Left Scapula', 'Right Scapula',
        'Left Lung', 'Right Lung', 'Left Hilus Pulmonis', 'Right Hilus Pulmonis',
        'Heart', 'Aorta', 'Facies Diaphragmatica', 'Mediastinum',  'Weasand', 'Spine']
    
    n_classes = len(classes)
    model = pspnet(n_classes)
    model_path = '.pspnet_chestxray_best_model_4.pkl'
    state = convert_state_dict(torch.load(model_path)["model_state"])
    model.load_state_dict(state)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.eval()
    model.to(device)
    print('model loaded!!')

    img = cv2.imread('demo.png', 1)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
    img = Transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)

    outputs = model(img)
    pred = outputs.data.cpu().numpy()
    pred = 1 / (1 + np.exp(-pred))  # sigmoid
    pred[pred < 0.5] = 0
    pred[pred > 0.5] = 1
    cv2.imwrite('demo_left_lung.png', pred[0, 4] * 255)

if __name__ == "__main__":
    main()
     
