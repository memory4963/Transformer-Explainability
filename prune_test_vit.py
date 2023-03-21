import torch
import torchvision
import torchvision.transforms as transforms
from baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
import json
import sys


if __name__ == '__main__':
    keep_heads = int(sys.argv[1])
    bsz = int(sys.argv[2])
    direction = bool(sys.argv[3])
    blk_info = json.load(open('block_info.json'))
    model = vit_LRP(pretrained=True).cuda()
    for i, blk in enumerate(model.blocks):
        mask = torch.zeros(12)
        idx = torch.tensor(blk_info[str(i)][::-1][:keep_heads] if direction else blk_info[str(i)][:keep_heads])
        mask[idx] = 1.
        blk.attn.attn_mask = mask.cuda()

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    val_set = torch.utils.data.DataLoader(
        torchvision.datasets.ImageNet('imagenet', split='val', transform=transform),
        batch_size=bsz,
        shuffle=False,
        num_workers=4
    )

    model.eval()
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(val_set):
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        # evaluate model here
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(f'{i}/{len(val_set)}, acc: {correct/total}', end='\r')
    print('Accuracy of the network on the 50000 val images: %.4f %%' % (
            100 * correct / total))

