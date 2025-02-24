import time
import torch
from timm import create_model
from arch.SFA.SFANet import SFANet

# 모델 생성 및 파라미터 수 측정 함수
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

mobilevit_s = create_model('mobilevit_s', num_classes=2, pretrained=True)
sfanet = SFANet(2, True)

mobilevit_s_params = count_parameters(mobilevit_s)
sfanet_params = count_parameters(sfanet)

print(f"MobileViT-S Parameters: {mobilevit_s_params / 1e6:.2f}M")
print(f"SFANet Parameters: {sfanet_params / 1e6:.2f}M")

# 추론 속도 비교
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mobilevit_s.to(device)
sfanet.to(device)

# 입력 데이터: 1개 배치, 3채널, 512x512
dummy_input = torch.randn(1, 3, 512, 512).to(device)

# Warm-up (각 모델 10회 실행)
with torch.no_grad():
    for _ in range(10):
        _ = mobilevit_s(dummy_input)
        _ = sfanet(dummy_input)

# 각 모델의 추론 시간을 측정 (100회 반복 후 평균)
with torch.no_grad():
    # MobileViT-S 추론 시간 측정
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    for _ in range(100):
        _ = mobilevit_s(dummy_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    mobilevit_time = (time.time() - start_time) / 100

    # SFANet 추론 시간 측정
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    for _ in range(100):
        _ = sfanet(dummy_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    sfanet_time = (time.time() - start_time) / 100

print(f"MobileViT-S Inference Time: {mobilevit_time * 1000:.2f} ms per image")
print(f"SFANet Inference Time: {sfanet_time * 1000:.2f} ms per image")
