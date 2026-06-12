import torch, glob, numpy as np, os
from PIL import Image
from torchvision import transforms
from ugbnet_segmentation_improved import UGBSegNetImproved

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UGBSegNetImproved().to(device)
ckpt = torch.load('saved_models/UGB_SegNet_best_split.pth', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Checkpoint best_dice: {ckpt['best_dice']*100:.2f}%")

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

imgs, masks = [], []
for cls in ['benign', 'malignant']:
    d = os.path.join('breast_ultrasound_dataset/test', cls)
    for p in sorted(glob.glob(os.path.join(d,'*.png')) + glob.glob(os.path.join(d,'*.jpg'))):
        if '_mask' in p: continue
        base = os.path.splitext(p)[0]
        for ext in ['_mask.png','_mask.jpg']:
            if os.path.exists(base+ext):
                imgs.append(p); masks.append(base+ext); break

print(f"Test images found: {len(imgs)}")

dices = []
with torch.no_grad():
    for ip, mp in zip(imgs, masks):
        img = to_tensor(Image.open(ip).convert('RGB').resize((224,224))).unsqueeze(0).to(device)
        msk = np.array(Image.open(mp).convert('L').resize((224,224), Image.NEAREST))
        msk = (msk > 127).astype(float)
        out = model(img)
        prob = torch.sigmoid(out['main'])[0,0].cpu().numpy()
        pred = (prob > 0.5).astype(float)
        tp = (pred * msk).sum()
        dices.append(2*tp / (pred.sum() + msk.sum() + 1e-8))

print(f"Test Dice: {np.mean(dices)*100:.2f}%")
print(f"Test images evaluated: {len(dices)}")