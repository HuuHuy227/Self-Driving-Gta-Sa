from torchvision import transforms
import torch

class ProcessImage():
    def __init__(self, img_size, model, DEVICE):
        self.img_size = img_size
        self.class_map = {0: 'backward', 1: 'backward_left', 2: 'backward_right', 3: 'forward',
                          4: 'forward_left', 5: 'forward_right', 6: 'left', 7: 'right'}
        self.action_map = {'backward': ['s'], 'backward_left': ['a','s'], 'backward_right': ['d','s'],
                           'forward': ['w'], 'forward_left': ['w','a'], 'forward_right': ['w','d'], 
                           'left': ['a'], 'right': ['d']}
        self.model = model
        self.device = DEVICE
    
    def transform_img(self, img):
        transform_ops = transforms.Compose([
            transforms.Resize(size = self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    
        return transform_ops(img)

    def predict(self, img):
        
        with torch.no_grad():
            self.model.eval()
            img = self.transform_img(img)
            img = img.unsqueeze(0)
            logits = self.model(img.to(self.device))
            res = logits.argmax(dim=1)
        ans = self.class_map[res.item()]
        print(ans)
        action = self.action_map[ans]
        return action