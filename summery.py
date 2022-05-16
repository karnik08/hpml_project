from torchvision import models
from torchsummary import summary
from model import net_G, net_D
import argparse
import torch

parser = argparse.ArgumentParser()
args = parser.parse_args()
if torch.cuda.is_available():
    device='cuds:0'
else:
    device='cpu'
G=net_G(args).to(device)
D=net_D(args).to(device)
# from torchsummary import summary
print('Generator Model Summery')
summary(G, input_size=(1,200),batch_size=1, device=device)

print('Discriminator Model Summery')
summary(D, input_size=(32, 32, 32),batch_size=1, device=device)