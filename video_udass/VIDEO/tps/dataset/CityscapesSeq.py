import numpy as np
import torch

from ADVENT.advent.utils.serialization import json_load
from tps.dataset.base_dataset import BaseDataset

class CityscapesSeqDataSet(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 set='val',
                 max_iters=None,
                 crop_size=(321, 321), 
                 mean=(128, 128, 128),
                 load_labels=True,
                 info_path='', 
                 labels_size=None, 
                 interval=1):
        super().__init__(root, list_path, set, max_iters, crop_size, labels_size, mean)
        self.load_labels = load_labels
        self.info = json_load(info_path)
        self.class_names = np.array(self.info['label'], dtype=np.str_)
        self.mapping = np.array(self.info['label2train'], dtype=np.int64)
        self.map_vector = np.zeros((self.mapping.shape[0],), dtype=np.int64)
        self.interval = interval
        for source_label, target_label in self.mapping:
            self.map_vector[source_label] = target_label

    def get_metadata(self, name):
        img_file = self.root / 'leftImg8bit_sequence' / self.set / name
        label_name = name.replace("leftImg8bit", "gtFine_labelIds")
        label_file = self.root / 'gtFine' / self.set / label_name
        return img_file, label_file

    def map_labels(self, input_):
        return self.map_vector[input_.astype(np.int64, copy=False)]

    def __getitem__(self, index):

        img_file, label_file, name_cf = self.files[index]
        label = self.get_labels(label_file)
        label = self.map_labels(label).copy()
        frame_cf = int(name_cf.split('/')[-1].replace('_leftImg8bit.png','')[-6:])

        # current 
        name_cf = name_cf
        file_cf = img_file
        d = self.get_image(file_cf)
        image_d = self.get_image(file_cf)
        image_d = self.preprocess(image_d)

        name_kf = name_cf.replace(str(frame_cf).zfill(6) + '_leftImg8bit.png', str(frame_cf - 1).zfill(6) + '_leftImg8bit.png')
        file_kf = self.root / 'leftImg8bit_sequence' / self.set / name_kf
        c = self.get_image(file_kf)
        image_c = self.get_image(file_kf)
        image_c = self.preprocess(image_c)

        # previous 
        name_kf = name_cf.replace(str(frame_cf).zfill(6) + '_leftImg8bit.png', str(frame_cf - self.interval).zfill(6) + '_leftImg8bit.png')
        file_kf = self.root / 'leftImg8bit_sequence' / self.set / name_kf
        b = self.get_image(file_kf) 
        image_b = self.get_image(file_kf)
        image_b = self.preprocess(image_b)

        name_kf = name_cf.replace(str(frame_cf).zfill(6) + '_leftImg8bit.png', str(frame_cf - self.interval - 1).zfill(6) + '_leftImg8bit.png')
        file_kf = self.root / 'leftImg8bit_sequence' / self.set / name_kf
        a = self.get_image(file_kf)
        image_a = self.get_image(file_kf)
        image_a = self.preprocess(image_a)

        frames = torch.tensor([frame_cf, frame_cf - 1, frame_cf - self.interval, frame_cf - self.interval - 1])
        
        if self.set == 'train':
            return image_d.copy(), image_c.copy(), image_b.copy(), image_a.copy(), d.transpose(2, 0, 1), label, name_cf, frames
        else:
            return image_d.copy(), label, image_c.copy(), image_b.copy(), image_a.copy(), name_cf