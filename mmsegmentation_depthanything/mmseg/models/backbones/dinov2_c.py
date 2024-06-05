import torch
from mmengine.model import BaseModule
from torch import nn

from mmseg.registry import MODELS
from timm.models.layers import trunc_normal_, DropPath
import torch.nn.functional as F
import clip

weather_texts = ["rainy weather, rain", "snowy weather, snow", "foggy weather, fog",
                      "clear weather, clear", "sunny weather, sun", "cloudy weather, cloud",
                      "overcast weather, overcast clouds", "partly cloudy weather, partly cloudy",
                      "misty weather, mist", "hazy weather, haze", "downpour weather, downpour rain",
                      "blizzard weather, blizzard", "precipitation weather, precipitation"]
class_texts = ['building', 'structure', 'road', 'sky', 'stone',
               'terrain-vegetation', 'terrain-other', 'terrain-snow', 'tree',]
weather_texts = ['building', 'structure', 'road', 'sky', 'stone',
                 'terrain-vegetation', 'terrain-other', 'terrain-snow', 'tree',] + weather_texts

def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()

    raise NotImplementedError(f'build_act_layer does not support {act_layer}')

class MLPLayer(nn.Module):
    r""" MLP layer of InternImage
    Args:
        in_features (int): number of input features
        hidden_features (int): number of hidden features
        out_features (int): number of output features
        act_layer (str): activation layer
        drop (float): dropout rate
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer='GELU',
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_act_layer(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        # for name, param in self.named_parameters():
        #     if 'weight' in name:
        #         nn.init.xavier_uniform_(param)
        #     elif 'bias' in name:
        #         nn.init.zeros_(param)


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)

class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)

def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


class DownsampleLayer(nn.Module):
    r""" Downsample layer of InternImage
    Args:
        channels (int): number of input channels
        norm_layer (str): normalization layer
    """

    def __init__(self, channels, norm_layer='LN'):
        super().__init__()
        self.conv = nn.Conv2d(channels,
                              2 * channels,
                              kernel_size=3,
                              stride=2,
                              padding=1,
                              bias=False)
        self.norm = build_norm_layer(2 * channels, norm_layer,
                                     'channels_first', 'channels_last')

    def forward(self, x):
        x = self.conv(x.permute(0, 3, 1, 2))
        x = self.norm(x)
        return x

class StemLayer(nn.Module):
    r""" Stem layer of InternImage
    Args:
        in_chans (int): number of input channels
        out_chans (int): number of output channels
        act_layer (str): activation layer
        norm_layer (str): normalization layer
    """

    def __init__(self,
                 in_chans=3,
                 out_chans=96,
                 act_layer='GELU',
                 norm_layer='BN'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans,
                               out_chans // 2,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm1 = build_norm_layer(out_chans // 2, norm_layer,
                                      'channels_first', 'channels_first')
        self.act = build_act_layer(act_layer)
        self.conv2 = nn.Conv2d(out_chans // 2,
                               out_chans,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm2 = build_norm_layer(out_chans, norm_layer, 'channels_first',
                                      'channels_last')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x

class CrossAttention(nn.Module):
    r""" Cross Attention Module
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 8
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: False.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        attn_head_dim (int, optional): Dimension of attention head.
        out_dim (int, optional): Dimension of output.
    """

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 attn_head_dim=None,
                 out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        assert all_head_dim == dim

        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.k_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, k=None, v=None):
        B, N, C = x.shape
        N_k = k.shape[1]
        N_v = v.shape[1]

        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = self.k_bias
            v_bias = self.v_bias

        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = q.reshape(B, N, 1, self.num_heads,
                      -1).permute(2, 0, 3, 1,
                                  4).squeeze(0)  # (B, N_head, N_q, dim)

        k = F.linear(input=k, weight=self.k.weight, bias=k_bias)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1,
                                                             4).squeeze(0)

        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1,
                                                             4).squeeze(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B, N_head, N_q, N_k)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class AttentiveBlock(nn.Module):
    r"""Attentive Block
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 8
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: False.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop (float, optional): Dropout rate. Default: 0.0.
        attn_drop (float, optional): Attention dropout rate. Default: 0.0.
        drop_path (float | tuple[float], optional): Stochastic depth rate.
            Default: 0.0.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm.
        attn_head_dim (int, optional): Dimension of attention head. Default: None.
        out_dim (int, optional): Dimension of output. Default: None.
    """

    def __init__(self,
                 dim,
                 num_heads,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer="LN",
                 attn_head_dim=None,
                 out_dim=None):
        super().__init__()

        self.norm1_q = build_norm_layer(dim, norm_layer, eps=1e-6)
        self.norm1_k = build_norm_layer(dim, norm_layer, eps=1e-6)
        self.norm1_v = build_norm_layer(dim, norm_layer, eps=1e-6)
        self.cross_dcn = CrossAttention(dim,
                                        num_heads=num_heads,
                                        qkv_bias=qkv_bias,
                                        qk_scale=qk_scale,
                                        attn_drop=attn_drop,
                                        proj_drop=drop,
                                        attn_head_dim=attn_head_dim,
                                        out_dim=out_dim)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        # for name, param in self.named_parameters():
        #     if 'weight' in name:
        #         nn.init.xavier_uniform_(param)
        #     elif 'bias' in name:
        #         nn.init.zeros_(param)

    def forward(self,
                x_q,
                x_kv,
                pos_q,
                pos_k,
                bool_masked_pos,
                rel_pos_bias=None):
        x_q = self.norm1_q(x_q + pos_q)
        x_k = self.norm1_k(x_kv + pos_k)
        x_v = self.norm1_v(x_kv)

        x = self.cross_dcn(x_q, k=x_k, v=x_v)

        return x

@MODELS.register_module()
class DINOv2(nn.Module):
    """Use DINOv2 pre-trained models
    """

    def __init__(self, version='large', freeze=False, load_from=None):
        super().__init__()
        
        if version == 'large':
            self.dinov2 = torch.hub.load('/home/zhangxd/project/mmsegmentation_depthanything_1/torchhub/facebookresearch_dinov2_main', 'dinov2_vitl14', source='local', pretrained=False)
        else:
            raise NotImplementedError

        if load_from is not None:
            d = torch.load(load_from, map_location='cpu')
            new_d = {}
            for key, value in d.items():
                if 'pretrained' in key:
                    new_d[key.replace('pretrained.', '')] = value
            self.dinov2.load_state_dict(new_d)
        
        self.freeze = freeze

        self.device= "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load('/home/zhangxd/project/mmsegmentation_depthanything_8/clip_model/ViT-B-32.pt', device=self.device, jit=False)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_text = clip.tokenize([weather for weather in weather_texts]).to(self.device)
        self.clip_text_encoded = self.clip_model.encode_text(self.clip_text)
        self.clip_mlp = MLPLayer(512,len(weather_texts),len(weather_texts),drop=0.2)
        in_chans = 3
        channels = 64
        act_layer = 'GELU'
        norm_layer = 'LN'
        self.patch_embed = StemLayer(in_chans=in_chans,
                                     out_chans=channels,
                                     act_layer=act_layer,
                                     norm_layer=norm_layer)
        self.pos_drop = nn.Dropout(p=0.)
        self.downsample = DownsampleLayer(channels=channels, norm_layer=norm_layer)

        self.query_size = [56, 28, 14, 7]
        # self.query_size = 32
        self.query_mlp_0 = nn.Linear(3136, 1024, dtype=torch.float32, device=self.device)
        self.query_mlp_1 = nn.Linear(784, 1024, dtype=torch.float32, device=self.device)
        self.query_mlp_2 = nn.Linear(196, 1024, dtype=torch.float32, device=self.device)
        self.query_mlp_3 = nn.Linear(49, 1024, dtype=torch.float32, device=self.device)
        self.query_mlp_list = [self.query_mlp_0, self.query_mlp_1, self.query_mlp_2, self.query_mlp_3]
        self.crossattention_0 = AttentiveBlock(1024, 8, out_dim=3136, drop=0.1, attn_drop=0.1,drop_path=0.1)
        self.crossattention_1 = AttentiveBlock(1024, 8, out_dim=784, drop=0.1, attn_drop=0.1,drop_path=0.1)
        self.crossattention_2 = AttentiveBlock(1024, 8, out_dim=196, drop=0.1, attn_drop=0.1,drop_path=0.1)
        self.crossattention_3 = AttentiveBlock(1024, 8, out_dim=49, drop=0.1, attn_drop=0.1,drop_path=0.1)
        self.crossattention_list = [self.crossattention_0, self.crossattention_1, self.crossattention_2, self.crossattention_3]

    def forward(self, inputs):
        # print('mmseg/models/backbones/dinov2/forward')
        # print('mmseg/models/backbones/dinov2/forward')
        # print('mmseg/models/backbones/dinov2/forward')
        # print('mmseg/models/backbones/dinov2/forward')
        # print('mmseg/models/backbones/dinov2/forward')
        # print('self.freeze: ', self.freeze)
        B, _, h, w = inputs.shape

        if h != 224 or w != 224:
            resized_inputs = F.interpolate(inputs.clone(), size=(224, 224), mode='bilinear', align_corners=False)
            clip_img_encoded = self.clip_model.encode_image(resized_inputs)
        else:
            clip_img_encoded = self.clip_model.encode_image(inputs)
        clip_img_encoded = clip_img_encoded.float()
        clip_text_encoded = self.clip_text_encoded.float()

        adverse_weights = self.clip_mlp(clip_img_encoded).unsqueeze(2).repeat(1, 1, 512)
        weathered_features = clip_text_encoded.unsqueeze(0).repeat(inputs.shape[0], 1, 1)
        weathered_features = weathered_features * adverse_weights
        weathered_features = torch.sum(weathered_features, dim=1, keepdim=True)

        # Step 5: Concatenate the image encodings with the summed text features
        clip_vector = torch.cat([clip_img_encoded.unsqueeze(1), weathered_features], dim=2)
        x = self.patch_embed(inputs)
        x = self.pos_drop(x)
        clip_vector = clip_vector.type_as(x)



        if self.freeze:
            with torch.no_grad():
                features = self.dinov2.get_intermediate_layers(inputs, 4)
        else:
            features = self.dinov2.get_intermediate_layers(inputs, 4)

        outs = []
        # for i, feature in features:
        for index, feature in enumerate(features):
            # print(index)
            C = feature.shape[-1]
            feature = feature.permute(0, 2, 1).reshape(B, C, h // 14, w // 14).contiguous()
            queries = feature
            if queries.shape[2] != self.query_size[index]:
                queries = F.adaptive_avg_pool2d(queries, (self.query_size[index], self.query_size[index]))
            queries = queries.view(queries.shape[0], queries.shape[1], -1)
            queries = self.query_mlp_list[index](queries)
            queries = self.crossattention_list[index](queries, clip_vector.unsqueeze(1), 0, 0, 0)
            queries = queries.view(queries.shape[0], queries.shape[1], self.query_size[index], self.query_size[index])
            if queries.shape[2] != feature.shape[2]:
                queries = F.interpolate(queries, (feature.shape[2], feature.shape[3]))
            feature = feature + queries

            outs.append(feature)
        
        return outs
