from torch import nn
from functools import partial

class SE_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, H, W):
        b, n, c = x.shape
        x = x.transpose(1,2).reshape(b, -1, H, W)
        # b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x_ = x * y.expand_as(x)
        return x_.flatten(2).transpose(1,2)

class SE_Block_S(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE_Block_S, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.fc = nn.Sequential(
            # nn.Linear(channel, channel // reduction, bias=False),
            nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel, bias=False),
            nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, bias=False),
            nn.Sigmoid()
            # nn.Softmax(dim=2)
        )

    def forward(self, x, H, W):
        b, n, c = x.shape
        x_ = x.transpose(1,2).reshape(b, -1, H, W)
        # # b, c, _, _ = x.size()
        x_ = self.conv(x_)
        y = self.fc(x_).flatten(2).transpose(1,2)
        # y = self.fc(y)
        return x * y
        x_ = x * y.expand_as(x)
        return x_

class Feature_Enhance(nn.Module):
    def __init__(self, in_dim, norm_layer=partial(nn.LayerNorm, eps=1e-6)) -> None:
        super().__init__()
        ###################Branch one###################
        self.ln1 = norm_layer(in_dim)
        self.act1 = nn.GELU()
        self.linear1 = nn.Linear(in_dim, in_dim)    # 考虑多个Linear或拆成多组分别linear（类似self-attention）
        ###################Branch two###################
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_dim, in_dim, 3, 1, 1, groups=in_dim)
    
        # self.bn4all = nn.BatchNorm2d(in_dim)  # 不用softmax得到概率值，而是类似卷积或Linaer去学习到每个特征点的权值，然后用BN来稳定
    def forward(self, x, H, W):
        """插在Transformer Encoder的MLP后面

        Args:
            x (Tensor): BxLxD
        """
        eps = 1e-20
        B, L, D = x.shape
        # x = x.transpose(1,2).reshape(B, -1, H, W)
        # x_c = self.ln1(x.flatten(2).transpose(1,2))     # B×L×D
        x_c = self.ln1(x)     # B×L×D
        x_c = self.act1(x_c)
        x_c = self.linear1(x_c)
        x = x.transpose(1,2).reshape(B, -1, H, W)
        # for atto+sci
        # attn_channel = x_c.transpose(1,2).reshape(B, -1, H, W)
        attn_channel = nn.Softmax(dim=1)(x_c)
        attn_channel = attn_channel.transpose(1,2).reshape(B, -1, H, W)

        x_s = self.bn1(x)
        x_s = self.relu(x_s)
        x_s = self.conv1(x_s)
        attn_spatial = nn.Softmax(dim=1)(x_s)
        attn = attn_channel * attn_spatial
        # x_attn = self.bn4all(x*attn)
        # x_ = x + x_attn
        x_sc = x*attn
        x_ = x + x_sc
        # x_ = x*attn
        # return x_.flatten(2).transpose(1,2)
        return x_.flatten(2).transpose(1,2)