import numpy as np
import torchvision.transforms as transforms
from baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
from dataset.image_loader import ImageFolder
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import sys
sys.path.append('../coresets')
import coresets

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
dataset = ImageFolder('imagenet/val/n02133161', transform, normalize, 14)
loader = DataLoader(dataset, 32, False, pin_memory=True)

model = vit_LRP(pretrained=True).cuda()
model.eval()

blk = model.blocks[11]
vs = [torch.empty([False, 64]).cuda() for _ in range(12)]
patches = [torch.empty([False, 3, 16, 16]) for _ in range(12)]
for img, entr_img, trans_img in tqdm(loader):
    img, entr_img = img.cuda(), entr_img.cuda()
    _ = model(img)
    # for blk in model.blocks:
    attn = blk.attn.get_attn()
    _, max_attn = attn.max(dim=3)
    v = blk.attn.get_v()
    for i, sample in enumerate(max_attn):
        for j, head in enumerate(sample):
            idx = torch.unique(head)
            vs[j] = torch.cat((vs[j], v[i, j, idx]))
            for k in idx:
                if k == 0:
                    patches[j] = torch.cat((patches[j], torch.zeros(1, 3, 16, 16)))
                k -= 1
                w = k//14
                h = k%14
                patches[j] = torch.cat((patches[j], trans_img[i:i+1, :, w*16:w*16+16, h*16:h*16+16]))

for i, v in tqdm(enumerate(vs)):
    # plt.subplot(3, 4, i+1)
    v = v.detach().cpu().numpy()
    km_coreset_gen = coresets.KMeansCoreset(v)
    C, w, idx = km_coreset_gen.generate_coreset(len(v)//100)
    tsne = TSNE(n_components=2, init='pca', perplexity=10)
    c = tsne.fit_transform(np.concatenate((C, v)))
    C_len = C.shape[0]
    plt.scatter(c[C_len:, 0], c[C_len:, 1], c='b')
    plt.scatter(c[:C_len, 0], c[:C_len, 1], c='r')
    for j, img in enumerate(patches[i][idx]):
        plt.subplot(10, 2, j+1)
        plt.imshow(img.numpy().transpose(1, 2, 0))
        if j == 19:
            break
    plt.savefig(f"head{i}.png")
    plt.show()
# scores = torch.empty([]).cuda()[False]
# for i, blk in enumerate(model.blocks):
#     scores = torch.cat((scores, blk.attn.get_head_score()))
#     # _, idx = torch.sort(blk.attn.get_head_score(), descending=True)
#     print(f"block {i}: {idx}")
# _, idx = torch.sort(scores, descending=True)
# print(f"scores: {scores}")
# print(f"order: {idx}")
