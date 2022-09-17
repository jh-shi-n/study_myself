import os
from torch.utils.data import Dataset
from PIL import Image


class fake_dataset(Dataset):
    # Check the dataset Path
    def read_data_set(self):
        all_img_files = []
        all_labels = []
        
        # Check the class name by using os.walk
        class_names = os.walk(self.dataset_path).__next__()[1]

        for index, class_name in enumerate(class_names):
            label = index
            img_dir = os.path.join(self.dataset_path, class_name)
            img_files = os.walk(img_dir).__next__()[2]

            for img_file in img_files:
                # Skip the other filesß
                if img_file.endswith('.jpg') is not True:
                    continue

                select_img = os.path.join(img_dir, img_file)
                img = Image.open(select_img).convert('L')
                if img is not None:
                    all_img_files.append(select_img)
                    all_labels.append(label)
        return all_img_files, all_labels, len(all_img_files), len(class_names)

    def __init__(self, path='PATH', transform=None):
        self.dataset_path = path
        self.img_files_path, self.labels, self.length, self.num_classes = self.read_data_set()
        self.transforms = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = Image.open(self.img_files_path[idx]).convert('L')
        id = self.labels[idx]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, id


if __name__ == "__main__":
    # TEST CODE
    idx = 1
    dataset = fake_dataset()
    img, id = dataset[idx]
    print("CHECK : whole number of images is {}".format(len(dataset)))
