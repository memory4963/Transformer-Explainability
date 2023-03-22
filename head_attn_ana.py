import torchvision.transforms as transforms
from baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
from dataset.image_loader import ImageFolder
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])
dataset = ImageFolder('imagenet/val', transform, normalize, 14)
loader = DataLoader(dataset, 32, False, pin_memory=True)

model = vit_LRP(pretrained=True).cuda()
model.eval()

for img, entr_img in tqdm(loader):
    img, entr_img = img.cuda(), entr_img.cuda()
    _ = model(img)
    for blk in model.blocks:
        attn = blk.attn.get_attn()
        sum_attn = attn.sum(dim=3)
        _, idx = sum_attn.max(dim=2)
        blk.attn.accum_head_score(torch.gather(entr_img, 1, idx).mean(0))

scores = torch.empty([]).cuda()[False]
for i, blk in enumerate(model.blocks):
    scores = torch.cat((scores, blk.attn.get_head_score()))
    # _, idx = torch.sort(blk.attn.get_head_score(), descending=True)
    print(f"block {i}: {idx}")
_, idx = torch.sort(scores, descending=True)
print(f"scores: {scores}")
print(f"order: {idx}")
