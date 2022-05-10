classes = ['dog', 'elephant', 'giraffe','guitar','horse','house','person']

# colab으로 하는게 낫겠음
class ArtPaintDataset(Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.df = df.reset_index()
        self.image_id = self.df.image_id
        self.labels = self.df.label
        self.transform = transform        
    
    def __len__(self):
        return len(self.df)

    def set_transform(self, transform):        
        self.transform = transform

    def __getitem__(self, idx):
        image_path = self.image_id[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image=np.array(image))['image']

        return {'image' : image, 'label' : label}