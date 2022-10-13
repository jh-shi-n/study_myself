import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class custom_dataset(Dataset):
    # Check the dataset Path
    def read_data_set(self):
        all_img_files = []
        all_labels = []

        class_names = os.walk(self.dataset_path).__next__()[1]

        for index, class_name in enumerate(class_names):
            label = index
            img_dir = os.path.join(self.dataset_path, class_name)
            img_files = os.walk(img_dir).__next__()[2]

            for img_file in img_files:
                # Skip the Non-image files
                if img_file.split(".")[-1] not in ['jpg', 'png']:
                    continue

                select_img = os.path.join(img_dir, img_file)
                # img = Image.open(select_img).convert(
                #     'RGB')  # change 1 channel into 3 channel
                img = Image.open(select_img)

                if img is not None:
                    all_img_files.append(select_img)
                    all_labels.append(label)
        return all_img_files, all_labels, len(all_img_files), len(class_names)

    def __init__(self, path, transform=None):
        self.dataset_path = path
        self.img_files_path, self.labels, self.length, self.num_classes = self.read_data_set()
        self.transforms = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Grayscale in 3 channel
        # img = Image.open(self.img_files_path[idx]).convert('RGB')

        # Grayscale in 1 channel
        img = Image.open(self.img_files_path[idx])

        id = self.labels[idx]
        img_path = self.img_files_path[idx]

        if self.transforms is not None:
            img = self.transforms(img)

        #img = np.expand_dims(np.array(img),axis=2)
        # print(np.array(img).shape)
        return img, id, img_path


if __name__ == "__main__":
    # TEST CODE
    idx = 1
    dataset = custom_dataset("PATH")
    img, id, img_path = dataset[idx]
    print(img_path)
