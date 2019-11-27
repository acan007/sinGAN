import torch

dis = torch.load('models/scale_factor=0.750000,alpha=10/0/netD.pth')
gen = torch.load('models/scale_factor=0.750000,alpha=10/0/netG.pth')

print(dis)
print(gen)

