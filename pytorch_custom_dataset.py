class Custom_Dataset(Dataset):
    def __init__(self, file_list, transforms=None):
        self.file_list = file_list
        self.transforms = transforms
    
    def __len__(self):
        return len(self.file_list)
      
    def __getitem__(self, idx):
        img = cv2. imread(self.file_list[idx])
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(im_rgb.transpose(1, 0, 2))
        
        if self.transforms is not None:
            img = self.transforms(img)
            
            
    return img
                              
                              
