import torch
from torch import nn
import torch.nn.functional as F
from hao.module import Module
from hao.transformer import MultiHeadAttention, AddNorm, PositionWiseFFN


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512):
        super().__init__()
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        self.conv = nn.LazyConv2d(num_hiddens, kernel_size=patch_size,
                                  stride=patch_size)

    def forward(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        return self.conv(X).flatten(2).transpose(1, 2)
    
class ViTMLP(nn.Module):
    def __init__(self, mlp_num_hiddens, mlp_num_outputs, dropout=0.5):
        super().__init__()
        self.dense1 = nn.LazyLinear(mlp_num_hiddens)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.LazyLinear(mlp_num_outputs)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.dense2(self.dropout1(self.gelu(self.dense1(x)))))


class ViTBlock(nn.Module):
    def __init__(self, num_hiddens, norm_shape, mlp_num_hiddens,
                 num_heads, dropout, use_bias=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(norm_shape)
        self.attention = MultiHeadAttention(num_hiddens, num_heads, dropout, use_bias)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens, dropout)

    def forward(self, X, valid_lens=None):
        X = X + self.attention(*([self.ln1(X)] * 3), valid_lens)
        return X + self.mlp(self.ln2(X))


class ViTEncoder(nn.Module):
    """Vision Transformer."""
    def __init__(self, img_size, patch_size, num_hiddens, mlp_num_hiddens,
                 num_heads, num_blks, emb_dropout, blk_dropout, use_bias=False):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, num_hiddens)
        num_steps = self.patch_embedding.num_patches
        # Positional embeddings are learnable
        self.pos_embedding = nn.Parameter(torch.randn(1, num_steps, num_hiddens))
        self.dropout = nn.Dropout(emb_dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f"{i}", ViTBlock(
                num_hiddens, num_hiddens, mlp_num_hiddens,
                num_heads, blk_dropout, use_bias))

    def forward(self, X):
        X = self.patch_embedding(X) 
        X = self.dropout(X + self.pos_embedding)
        return self.blks(X) # (B, P, C)
    

class DecoderBlock(nn.Module):
    def __init__(self, num_hiddens, mlp_num_hiddens, num_heads, dropout):
        super().__init__()
        self.attention1 = MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.attention2 = MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(num_hiddens, mlp_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(num_hiddens, dropout)

    def forward(self, X, enc_outputs, dec_valid_lens=None, enc_valid_lens=None):
        key_values = X
        # 自注意力
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # 编码器－解码器注意力。
        # enc_outputs的开头:(batch_size,num_steps,num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z))


class TransformerDecoder(nn.Module):
    def __init__(self, img_size, patch_size, num_hiddens, num_heads, mlp_num_hiddens, num_layers, blk_dropout, num_classes=2):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_hiddens = num_hiddens
        self.num_classes = num_classes

        self.blks = nn.Sequential()
        self.head = nn.Linear(num_hiddens, num_classes)
        for i in range(num_layers):
            self.blks.add_module(f"{i}", DecoderBlock(num_hiddens, mlp_num_hiddens, num_heads, blk_dropout))

    def forward(self, X, enc_outputs):
        for i, blk in enumerate(self.blks):
            X = blk(X, enc_outputs) # (B, P, num_hiddens)
        X = self.head(X) # (B, P, num_classes)
        B, P, C = X.shape
        H = W = int(P ** 0.5)
        X = X.permute(0, 2, 1).reshape(B, C, H, W)  # 变换形状
        X = F.interpolate(X, size=(self.img_size, self.img_size), mode='bilinear', align_corners=True)
    
        return X #(B, H, W, num_classes)

class Segmenter(Module):
    def __init__(self, img_size=128, patch_size=16, num_hiddens=512, mlp_num_hiddens=1024, num_heads=8, num_blks=6, emb_dropout=0.1, blk_dropout=0.1, num_classes=2, lr=1e-4, smooth=1e-5):
        super().__init__(plot_train_per_epoch=2, plot_valid_per_epoch=1)
        self.save_hyperparameters()
        # 编码器：ViT 部分
        self.encoder = ViTEncoder(img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads, num_blks, emb_dropout, blk_dropout)
        # Transformer 解码器
        self.decoder = TransformerDecoder(img_size, patch_size, num_hiddens, num_heads, mlp_num_hiddens, num_blks, blk_dropout, num_classes)

    def forward(self, X):
        enc_outputs = self.encoder(X)  # (B, P, C)
        # print(enc_outputs.shape)
        X = self.decoder(enc_outputs, enc_outputs)
        return X  # 输出 (B, num_classes, H, W)
    
    def loss(self, preds, targets):
        """
        preds: (B, num_classes, H, W) - softmax 后的概率
        targets: (B, 1, H, W) - 真实标签（类别索引）
        """
        preds = torch.softmax(preds, dim=1)  # (B, num_classes, H, W)
        
        # 1. 去掉通道维度，使其变成 (B, H, W)
        targets = targets.squeeze(1)  # (B, H, W)
        
        # 2. 进行 one-hot 编码，结果是 (B, H, W, num_classes)
        targets_one_hot = torch.nn.functional.one_hot(targets.long(), num_classes=preds.shape[1])
        
        # 3. 调整维度顺序，使其变成 (B, num_classes, H, W)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).to(preds.dtype)  
        
        # print('1', preds.shape)          # (B, num_classes, H, W)
        # print('2', targets_one_hot.shape)  # (B, num_classes, H, W)

        intersection = torch.sum(preds * targets_one_hot, dim=[0, 2, 3])
        union = torch.sum(preds, dim=[0, 2, 3]) + torch.sum(targets_one_hot, dim=[0, 2, 3])

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 每 10 个 epoch 学习率衰减 0.1
        return optimizer, scheduler

    
