import torch
import torch.nn as nn
import timm
from thop import profile

from arch.architecture import DSDBNet  # Modified import statement to absolute

device = 'cuda'
backbone = 'mobilevit_s.cvnets_in1k'

# Both inputs have 3 channels
spect_chans = 3
spect_encoder = timm.create_model(backbone, in_chans=spect_chans, num_classes=2, pretrained=True)    
png_encoder = timm.create_model(backbone, in_chans=3, num_classes=2, pretrained=True)

model = DSDBNet(spect_encoder=spect_encoder, png_encoder=png_encoder, num_classes=2, feature_fusion='concat', global_pool=True, spect_input_size=(1, spect_chans, 224, 224), png_input_size=(1, 3, 224, 224)).to(device)
# model = spect_encoder.to(device)
dummy_spect = torch.randn(1, spect_chans, 224, 224).to(device)
dummy_png = torch.randn(1, 3, 224, 224).to(device)

flops, params = profile(model, inputs=(dummy_spect, dummy_png))
# flops, params = profile(model, inputs=(dummy_spect, ))
print(f"FLOPs: {flops / 1e9:.2f} G")
print(f"Params: {params / 1e6:.2f} M")

