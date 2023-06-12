from PIL import Image
import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
import json
import pathlib
import unittest
from typing import Tuple

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
transformations_train = transforms.Compose([transforms.ToPILImage(),
                                transforms.RandomChoice([
                                    transforms.ColorJitter(brightness=(0.80, 1.20)),
                                    transforms.RandomGrayscale(p = 0.25)
                                    ]),
                                transforms.RandomHorizontalFlip(p = 0.25),
                                transforms.RandomRotation(25),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = mean, std = std),
                                ])

class VOCSegDataset(datasets.VOCSegmentation):
    def __init__(self, root, image_set, transforms=None, output_size=None, malicious=False, **kwargs):
        super().__init__(root=root, image_set=image_set, transforms=transforms, **kwargs)
        if output_size is not None:
            self.resize = torchvision.transforms.Resize(output_size, interpolation=InterpolationMode.NEAREST)
        else:
            self.resize = None
        self.malicious = malicious
        self.num_of_voc_classes = 21
    
    def set_only_use_class(self):
        drop_indices = self.__filter_None_classes__(self.masks)
        print(f"Drop {len(drop_indices)} images")
        self.images = [self.images[i] for i in range(len(self.images)) if i not in drop_indices]
        self.targets = [self.targets[i] for i in range(len(self.targets)) if i not in drop_indices]
    
    def __filter_None_classes__(self, masks):
        only_use_class = self.get_only_use_class()
        indices = []
        for i in range(len(masks)):
            # if has only background
            label = Image.open(masks[i])
            unique_class = np.unique(label, return_counts=False)
            # filter only_use_class
            unique_class = [c for c in unique_class if c in only_use_class]
            if len(unique_class) == 1 and unique_class[0] == 0:
                indices.append(i)
                continue
        return indices
    
    def __getitem__(self, idx):
        # image = cv2.cvtColor(cv2.imread(self.images[idx]), cv2.COLOR_BGR2RGB)
        image = Image.open(self.images[idx]).convert('RGB')
        label = Image.open(self.masks[idx])
    
        if self.transforms is not None:
            augmented = self.transforms(image=np.array(image), mask=np.array(label))
            image = augmented['image'] #/ 255# .float()
            label = augmented['mask'].long() # torch.Size([224, 224])
            label[label > 20] = 0
            # label = self.__set_only_use_class__(label)
            if self.resize is not None:
                label = self.resize(label.unsqueeze(0)).squeeze(0)
        
        if self.malicious:
            label = self.__shift_segmentation_mask__(label)
            
        return image, label
    
    def set_malicious(self, malicious):
        self.malicious = malicious

    def get_only_use_class(self):
        return [0, 1, 9, 12] # background, airplane, chair, dog 

    def __set_only_use_class__(self, label):
        only_use_class = self.get_only_use_class()
        for i in range(self.num_of_voc_classes):
            if i not in only_use_class:
                label[label == i] = 0
        for i in range(len(only_use_class)):
            label[label == only_use_class[i]] = i
        return label

    def __shift_segmentation_mask__(self, mask):
        shifted_mask = mask.clone() # 원본 마스크 복사

        # 픽셀 단위로 순회하면서 클래스 인덱스 변경
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                current_index = mask[i, j]
                
                # Background (클래스 인덱스 0)는 변경하지 않음
                if current_index == 0:
                    continue

                # 클래스 인덱스 변경 (1->2, 2->3, ... 19->20, 20->1)
                shifted_index = current_index + 1 if current_index < 20 else 1
                shifted_mask[i, j] = shifted_index

        return shifted_mask
    
# albumentations_transform = A.Compose([
#     A.Resize(256, 256),  # 이미지 크기 조정
#     A.RandomCrop(224, 224),  # 랜덤으로 이미지 자르기
#     A.HorizontalFlip(p=0.5),  # 50% 확률로 좌우 반전
#     A.Rotate(limit=45),  # 최대 45도까지 랜덤 회전
#     A.RandomBrightnessContrast(p=0.2),  # 20% 확률로 밝기와 명암 대비 조정
#     # A.OneOf([A.GaussNoise(var_limit=(10.0, 50.0)),  # 가우시안 노이즈 추가
#     #          A.GaussianBlur(blur_limit=3),  # 가우시안 블러 적용
#     #          A.MotionBlur(blur_limit=3)]),  # 모션 블러 적용
#     # A.OneOf([A.OpticalDistortion(distort_limit=1.0),  # 광학 왜곡 추가
#     #          A.GridDistortion(num_steps=5, distort_limit=1.),
#     #          A.ElasticTransform(alpha=3)]),  # 탄성 변형 추가
#     A.RandomContrast(limit=0.2),  # 랜덤한 명암 대비 조정
#     A.Normalize(),  # 이미지 정규화
#     ToTensorV2()
# ])

# def getVOCSegDatasets(output_size=None):
#     # Creating the dataset
#     train_dataset = VOCSegDataset(
#         root='~/.data/',
#         image_set='train',
#         year='2012',
#         download=False,
#         transforms=albumentations_transform,
#         output_size=output_size
#     )
#     valid_dataset = VOCSegDataset(
#         root='~/.data/',
#         image_set='val',
#         year='2012',
#         download=False,
#         transforms=albumentations_transform,
#         output_size=output_size
#     )
#     return train_dataset, valid_dataset


class mydataset(torch.utils.data.Dataset):
    def __init__(self, imgs, labels, train=False, verbose=False, transforms=None):
        self.img = imgs
        self.gt = labels
        self.train = train
        self.verbose = verbose
        self.aug = False
        self.transforms = transforms
        self.dataset = list(zip(self.img, self.gt))
        return
    def __len__(self):
        return len(self.img)
    def __getitem__(self, idx):
        img = self.img[idx]
        gt = self.gt[idx]
        # print(img.shape) # 3, 224, 224
        if self.transforms:
            img = self.transforms(img)
        idx = torch.tensor(0)
        
        # total_classes = self.get_voc_total_classes()
        # used_classes = self.get_used_classes()
        
        
        # # for c in enumerate(total_classes):
        # #     for j, uc in enumerate(used_classes):
        # #         gt[gt == uc] = j
        # i = 1
        # for class_index in range(1, 21):
        #     if class_index not in [1, 9, 12]: # airplane, chair, dog
        #         gt[gt==class_index] = 0
        #     else:
        #         gt[gt==class_index] = i
        #         i += 1
            
        # return img, gt, idx
        return img, gt
    
    def get_voc_total_classes(self):
        return { 0: "background", 1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle", 6: "bus", 7: "car", 8: "cat", 9: "chair", 10: "cow", 11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike", 15: "person", 16: "pottedplant", 17: "sheep", 18: "sofa", 19: "train", 20: "tvmonitor" }
    
    def get_used_classes(self):
        return { 0: "background", 1: "aeroplane", 9: "chair", 12: "dog" }
    
    def get_labels(self):
        return self.gt
    
class PascalVocSegmentationPartition():
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.imagesize = (args.input_size, args.input_size)
        self.output_size = (args.output_size, args.output_size)
        self.train_dataset = VOCSegDataset( root='~/.data/', image_set='train', year='2012', download=False, transforms=self.get_train_transform(), output_size=self.output_size )
        self.test_dataset = VOCSegDataset( root='~/.data/', image_set='val', year='2012', download=False, transforms=self.get_train_transform(), output_size=self.output_size )
        self.malicious = np.random.choice(range(self.args.num_clients), size=int(self.args.malicious), replace=False)
        print(f"malicious clients: {self.malicious}")
    
    def get_train_transform(self):
        albumentations_transform = A.Compose([
            A.PadIfNeeded(min_height=256, min_width=256),
            A.RandomCrop(self.args.input_size, self.args.input_size),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        return albumentations_transform
    
    def load_partition(self, partition=-1):
        # load split files
        split_path = pathlib.Path(f"/home/suncheol/code/FedTest/0_FedMHAD_Seg/splitfile/{self.args.dataset}_{self.args.num_clients}_clients")
        if split_path.exists():
            if self.args.alpha > 0:
                print("dirichlet")
                with open(split_path / f'dirichlet_{self.args.alpha}_for_{self.args.num_clients}_clients', "r") as f:
                    split_data_index_dict = json.load(f)
                    split_data_index_dict = {int(k):v for k,v in split_data_index_dict.items()}
            else:
                print("iid")
                with open(split_path / f'iid_for_{self.args.num_clients}_clients', "r") as f:
                    split_data_index_dict = json.load(f)
                    split_data_index_dict = {int(k):v for k,v in split_data_index_dict.items()}
        
        test_index_path = split_path / "test_index"
        if test_index_path.exists():
            with open(test_index_path, "r") as f:
                test_index = json.load(f)
                if test_index is not None:
                    test_index = test_index["test"]
        
        if partition in self.malicious:
            self.train_dataset.set_malicious(True)
            
        if partition >= 0:
            train_dataset = torch.utils.data.Subset(self.train_dataset, split_data_index_dict[partition])
            # test_dataset = self.test_dataset
            # torch.utils.data.Subset(self.test_dataset, test_index) if test_index is not None else self.test_dataset
            n_test = int(len(self.test_dataset) / self.args.num_clients)
            test_dataset = torch.utils.data.Subset(self.test_dataset, range(partition * n_test, (partition + 1) * n_test))
        else:
            train_dataset = self.train_dataset
            test_dataset = self.test_dataset #torch.utils.data.Subset(self.test_dataset, test_index) if test_index is not None else self.test_dataset
            
        print(len(train_dataset), len(test_dataset))
        return train_dataset, test_dataset
            
    # def load_public_dataset(self):
    #     path = pathlib.Path.home().joinpath('.data', 'ImageNet')
    #     if not path.joinpath('train_images_1_20.npy').exists():
    #         public_imgs = np.load(path.joinpath('train_images.npy'))
    #         public_labels = np.load(path.joinpath('train_labels.npy'))
    #         index = np.random.choice(public_imgs.shape[0], int(public_imgs.shape[0]/20), replace=False)
    #         public_imgs = public_imgs[index]
    #         public_labels = public_labels[index]
    #         np.save(path.joinpath('train_images_1_20.npy'), public_imgs)
    #         np.save(path.joinpath('train_labels_1_20.npy'), public_labels)
    #     else :
    #         public_imgs = np.load(path.joinpath('train_images_1_20.npy'))
    #         public_labels = np.load(path.joinpath('train_labels_1_20.npy'))
    #     public_set = mydataset(public_imgs, public_labels)
    #     return public_set
    
    def load_public_dataset(self):
        path = pathlib.Path.home().joinpath('.data', 'MSCOCO')
        if not path.joinpath('coco_img_1_10.npy').exists():
            public_imgs = np.load(path.joinpath('coco_img.npy'))
            public_labels = np.load(path.joinpath('coco_label.npy'))
            index = np.random.choice(public_imgs.shape[0], int(public_imgs.shape[0]/10), replace=False)
            public_imgs = public_imgs[index]
            public_labels = public_labels[index]
            np.save(path.joinpath('coco_img_1_10.npy'), public_imgs)
            np.save(path.joinpath('coco_label_1_10.npy'), public_labels)
        else :
            public_imgs = np.load(path.joinpath('coco_img_1_10.npy'))
            public_labels = np.load(path.joinpath('coco_label_1_10.npy'))
        # random sampling 1/10 of the data
        # public_imgs = (public_imgs.transpose(0, 2, 3, 1)*255.0).round().astype(np.uint8)
        # print("size of public dataset: ", public_imgs.shape, "images")
        # public_imgs, public_labels = self.filter_images_by_label_type(self.args.task, public_imgs, public_labels)
        # public_dataset = mydataset(public_imgs, public_labels, transforms=transformations_train)
        print("size of public dataset: ", public_imgs.shape, "images")
        public_imgs, public_labels = self.filter_images_by_label_type(self.args.task, public_imgs, public_labels)
        public_dataset = mydataset(public_imgs, public_labels)
        return public_dataset

    def filter_images_by_label_type(self, task: str, imgs: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        print(f"filtering images by label type: {task}")
        if task == 'singlelabel':
            sum_labels = np.sum(labels, axis=1)
            index = np.where(sum_labels == 1)
            labels = labels[index]
            labels = np.argmax(labels, axis=1)
            imgs = imgs[index]
        elif task == 'multilabel_only':
            sum_labels = np.sum(labels, axis=1)
            index = np.where(sum_labels > 1)
            labels = labels[index]
            imgs = imgs[index]
        elif task == 'multilabel':
            pass
        return imgs, labels

class Test_PascalVocPartition(unittest.TestCase):
    def test_load_partition(self):
        args = argparse.Namespace()
        args.datapath = '~/.data'
        args.num_clients = 10
        args.alpha = 1.0
        args.task = 'multilabel'
        args.dataset = 'voc2012'
        args.batch_size = 16
        pascal = PascalVocSegmentationPartition(args)
        train_dataset, test_parition = pascal.load_partition(0)
        print(len(train_dataset), len(test_parition))
        train_dataset, test_parition = pascal.load_partition(1)
        print(len(train_dataset), len(test_parition))
        train_dataset, test_parition = pascal.load_partition(2)
        print(len(train_dataset), len(test_parition))
        train_dataset, test_parition = pascal.load_partition(-1)
        print(len(train_dataset), len(test_parition))  

        public_set = pascal.load_public_dataset()
        print(len(public_set))
        
        def get_labels(arr):
            unique_list = np.unique(arr)
            unique_list = unique_list[unique_list!=0]
            unique_list.sort()
            return list(unique_list)
        import time
        t = time.time()
        labels = [get_labels(train_dataset[i][1].numpy()) for i in range(len(train_dataset))]
        print(time.time() - t)
        print(labels)
if __name__ == '__main__':
    unittest.main()
# %%
