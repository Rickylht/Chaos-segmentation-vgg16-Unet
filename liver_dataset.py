from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms.functional as TF


class LiverDataset(Dataset):
    def __init__(self, root_dir):
        self.data_path = self.get_data_path(root_dir)

    def __getitem__(self, index):
        img_path, lbl_path = self.data_path[index]
        img = Image.open(img_path).convert('L')
        lbl = Image.open(lbl_path).convert('L')
        img = TF.to_tensor(img)
        lbl = TF.to_tensor(lbl)
        return img, lbl

    def __len__(self):
        return len(self.data_path)

    def get_data_path(self, root):
        data_path = []
        img_path = os.path.join(root, "img")
        lbl_path = os.path.join(root, "lbl")
        names = os.listdir(img_path)
        n = len(names)
        for i in range(n):
            img = os.path.join(img_path, names[i])
            lbl = os.path.join(lbl_path, names[i])
            data_path.append((img, lbl))
        return data_path
    
