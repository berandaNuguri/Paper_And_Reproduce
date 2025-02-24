import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .SFA import StegoScore
from ..longformer.diagonaled_mm_tvm import diagonaled_mm as diagonaled_mm_tvm, mask_invalid_locations
from ..longformer.sliding_chunks import sliding_chunks_matmul_qk, sliding_chunks_matmul_pv
from ..longformer.sliding_chunks import sliding_chunks_no_overlap_matmul_qk, sliding_chunks_no_overlap_matmul_pv

# ----- 공식 LongFormerSelfAttention (공식 코드 일부 발췌) -----
class LongformerSelfAttention(nn.Module):
    def __init__(self, config, layer_id):
        super(LongformerSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention heads (%d)" %
                             (config.hidden_size, config.num_attention_heads))
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size

        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        self.key = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)

        # 글로벌 어텐션용 (필요시)
        self.query_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.key_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.value_global = nn.Linear(config.hidden_size, self.embed_dim)

        self.dropout = config.attention_probs_dropout_prob

        self.layer_id = layer_id
        self.attention_window = config.attention_window[self.layer_id]
        self.attention_dilation = config.attention_dilation[self.layer_id]
        self.attention_mode = config.attention_mode
        self.autoregressive = config.autoregressive
        assert self.attention_window > 0
        assert self.attention_dilation > 0
        assert self.attention_mode in ['tvm', 'sliding_chunks', 'sliding_chunks_no_overlap']
        if self.attention_mode in ['sliding_chunks', 'sliding_chunks_no_overlap']:
            assert not self.autoregressive  # not supported
            assert self.attention_dilation == 1  # dilation is not supported

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        # 공식 코드를 그대로 사용. (자세한 내용은 공식 코드를 참고)
        # 여기서는 output_attentions=False인 경우 context_layer만 반환한다고 가정합니다.
        # 실제 구현에서는 공식 LongformerSelfAttention 코드에 따라 반환값이 튜플로 구성됩니다.
        hidden_states = hidden_states.transpose(0, 1)  # (seq_len, batch, embed_dim)
        seq_len, bsz, embed_dim = hidden_states.size()
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        q /= math.sqrt(self.head_dim)
        # 간단화를 위해 sliding_chunks 방식을 사용
        q = q.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)  # (batch, seq_len, num_heads, head_dim)
        k = k.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        attn_weights = sliding_chunks_matmul_qk(q, k, self.attention_window, padding_value=0)
        mask_invalid_locations(attn_weights, self.attention_window, self.attention_dilation, False)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = sliding_chunks_matmul_pv(attn_probs, v, self.attention_window)
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, bsz, embed_dim)
        context_layer = attn_output.transpose(0, 1).contiguous()  # (batch, seq_len, embed_dim)
        return (context_layer,)

# ----- Minimal LongformerConfig for our block -----
class MyLongformerConfig:
    def __init__(self,
                 hidden_size=512,
                 num_attention_heads=4,
                 attention_window=8,
                 attention_dilation=[1],
                 autoregressive=False,
                 attention_mode='sliding_chunks',
                 attention_probs_dropout_prob=0.1):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        # 공식 코드는 각 layer마다 window size를 리스트로 받습니다.
        self.attention_window = [attention_window]
        self.attention_dilation = attention_dilation
        self.autoregressive = autoregressive
        self.attention_mode = attention_mode
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

# ----- MyLongformerBlock: 공식 LongformerSelfAttention 기반 transformer block -----
class MyLongformerBlock(nn.Module):
    def __init__(self, config):
        super(MyLongformerBlock, self).__init__()
        # 한 층의 LongformerSelfAttention
        self.attention = LongformerSelfAttention(config, layer_id=0)
        # 간단한 feed-forward 네트워크
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size)
        )
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
    
    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        attn_output = self.attention(x, output_attentions=False)[0]
        x = x + attn_output
        x = self.norm1(x)
        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.norm2(x)
        return x

# ----- 기존의 ConvBlock 및 ChannelWiseStegoScore -----
class ConvBlock(nn.Module):
    """
    ConvBlock: Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm.
    residual_connection 인자가 True이면, 입력 x를 shortcut으로 더한 후 출력합니다.
    첫 번째 Conv2d에 stride 인자를 적용합니다.
    """
    def __init__(self, in_channels, out_channels, stride=1, residual_connection=False):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.residual_connection = residual_connection
        if self.residual_connection:
            # 만약 입력 차원이나 stride가 달라서 직접 덧셈이 불가능하면 1x1 conv로 차원 맞춤
            if in_channels != out_channels or stride != 1:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.shortcut = nn.Identity()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.residual_connection:
            out = out + self.shortcut(x)
        return out


class ChannelWiseStegoScore(nn.Module):
    """
    입력 이미지의 각 채널에 대해 StegoScore를 개별적으로 적용합니다.
    trainable_filter가 True이면 채널별로 독립적인 StegoScore 인스턴스를 생성하고,
    False이면 하나의 인스턴스를 공유하여 적용합니다.
    입력: (B, C, H, W) → 출력: (B, C, H, W)
    """
    def __init__(self, trainable_filter=True):
        super(ChannelWiseStegoScore, self).__init__()
        self.trainable_filter = trainable_filter
        if self.trainable_filter:
            self.stego_list = nn.ModuleList([StegoScore(trainable_filter=True) for _ in range(3)])
        else:
            self.stego = StegoScore(trainable_filter=False)
    
    def forward(self, x):
        outputs = []
        B, C, H, W = x.shape
        if self.trainable_filter:
            for c in range(C):
                x_c = x[:, c:c+1, :, :]
                x_c_score = self.stego_list[c](x_c)
                outputs.append(x_c_score)
        else:
            for c in range(C):
                x_c = x[:, c:c+1, :, :]
                x_c_score = self.stego(x_c)
                outputs.append(x_c_score)
        out = torch.cat(outputs, dim=1)
        return out

# ----- SFANet 모델 (공식 LongFormer 기반 Transformer 사용) -----
class SFANet(nn.Module):
    """
    SFANet은 입력에 대해 채널별 StegoScore를 적용한 후,
    ConvBlock들을 통해 특징을 추출하고,
    LongFormer 기반 Transformer 블록(MyLongformerBlock)을 통해 글로벌 정보를 반영합니다.
    마지막 Global Pooling 및 FC Layer로 Cover/Stego 2진 분류를 수행합니다.
    """
    def __init__(self, num_classes=2, residual_connection=True, trainable_filter=True, attention_window=8):
        super(SFANet, self).__init__()
        # 1. 초기 채널별 StegoScore 적용
        self.initial_stego = ChannelWiseStegoScore(trainable_filter=trainable_filter)

        # 2. Convolution Block들 (downsampling을 위해 stride 적용)
        self.block1 = ConvBlock(3, 64, stride=1, residual_connection=residual_connection)
        self.block2 = ConvBlock(64, 128, stride=2, residual_connection=residual_connection)
        self.block3 = ConvBlock(128, 256, stride=2, residual_connection=residual_connection)
        self.block4 = ConvBlock(256, 512, stride=2, residual_connection=residual_connection)
        # block4 출력: (B, 512, H/16, W/16)
        
        # 3. LongFormer 기반 Transformer 블록 적용
        # 먼저, flatten하여 (B, seq_len, hidden_dim) 형태로 변환
        # 예: 입력 이미지 224x224 → block4 출력: 512 x 14 x 14 → seq_len = 196
        # 구성용 config 생성
        config = MyLongformerConfig(hidden_size=512,
                                    num_attention_heads=4,
                                    attention_window=attention_window,
                                    attention_dilation=[1],
                                    autoregressive=False,
                                    attention_mode='sliding_chunks',
                                    attention_probs_dropout_prob=0.1)
        # 만약 여러 층으로 구성하려면 MyLongformerBlock을 depth 반복해서 쌓을 수 있음.
        # 여기서는 단일 층으로 구성합니다.
        self.longformer_block = MyLongformerBlock(config)
        
        # 4. 최종 Global Pooling 및 FC Layer (입력 채널 512)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes, bias=False)
    
    def forward(self, x):
        # 1. 초기 채널별 StegoScore 적용 (B, 3, H, W)
        x = self.initial_stego(x)

        # 2. ConvBlock들을 통한 특징 추출
        x = self.block1(x)         # (B, 64, H/2, W/2)
        x = self.block2(x)         # (B, 128, H/4, W/4)
        x = self.block3(x)         # (B, 256, H/8, W/8)
        x = self.block4(x)         # (B, 512, H/16, W/16)
        
        # 3. Flatten spatial dimensions and apply LongFormer block
        B, C, H, W = x.shape
        seq_len = H * W
        # 변환: (B, 512, H, W) → (B, seq_len, 512)
        x_seq = x.view(B, C, -1).permute(0, 2, 1)
        # LongFormer 블록 적용

        x_seq = self.longformer_block(x_seq)  # (B, seq_len, 512)
        # 복원: (B, seq_len, 512) → (B, 512, H, W)
        x = x_seq.permute(0, 2, 1).contiguous().view(B, C, H, W)
        
        # 4. Global Pooling 및 FC Layer로 최종 분류
        x_pool = self.global_pool(x).view(B, -1)
        logits = self.fc(x_pool)
        return logits
