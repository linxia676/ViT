import torch
from torch import nn
from hao.module import Classifier
from hao.transformer import MultiHeadAttention
from hao.CLIP import CLIP

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
        return self.dropout2(self.dense2(self.dropout1(self.gelu(
            self.dense1(x)))))


class ViTBlock(nn.Module):
    def __init__(self, num_hiddens, norm_shape, mlp_num_hiddens,
                 num_heads, dropout, use_bias=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(norm_shape)
        self.attention = MultiHeadAttention(num_hiddens, num_heads,
                                                dropout, use_bias)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens, dropout)

    def forward(self, X, valid_lens=None):
        X = X + self.attention(*([self.ln1(X)] * 3), valid_lens)
        return X + self.mlp(self.ln2(X))


class Instruction_ViT(Classifier):
    def __init__(self, img_size, patch_size, num_hiddens, mlp_num_hiddens,
                 num_heads, num_blks, emb_dropout, blk_dropout, texts, lr=0.1,
                 use_bias=False, num_classes=7):
        super().__init__()
        self.save_hyperparameters()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, num_hiddens)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hiddens))

        self.clip = CLIP()
        self.prompt_proj = nn.Linear(512,num_hiddens)

        self.prompt_tokens = self.prompt_proj(self.clip.encode_text(texts))[None, :]
        # self.prompt_tokens = self.prompt_proj(torch.rand(num_classes, 512)).unsqueeze(0)
        self.prompt_tokens = nn.Parameter(self.prompt_tokens)
        num_steps = self.patch_embedding.num_patches + 1 + num_classes  # Add the cls + prompt token

        # Positional embeddings are learnable
        self.pos_embedding = nn.Parameter(torch.randn(1, num_steps, num_hiddens))
        self.dropout = nn.Dropout(emb_dropout)
        self.final_norm = nn.LayerNorm(num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f"{i}", ViTBlock(
                num_hiddens, num_hiddens, mlp_num_hiddens,
                num_heads, blk_dropout, use_bias))
        self.head = nn.Sequential(nn.LayerNorm(num_hiddens),
                                  nn.Linear(num_hiddens, num_classes))

    def forward(self, X):
        X = self.patch_embedding(X)
        X = torch.cat((self.cls_token.expand(X.shape[0], -1, -1), X, 
                       self.prompt_tokens.expand(X.shape[0], -1, -1)), 1)
        X = self.dropout(X + self.pos_embedding)
        for blk in self.blks:
            X = blk(X)
        X = self.final_norm(X)
        # print(X)
        image_features, text_features = X[:, 0], X[:, -self.num_classes:]
        image_features = torch.nn.functional.normalize(image_features,p=2,dim=1) 
        text_features = torch.nn.functional.normalize(text_features,p=2,dim=2)
        # print(text_features)
        # image_features: [batch, embedding]
        # text_features: [batch, category, embedding]
        # output: [batch, category]
        similarity = torch.einsum('be,bce->bc', image_features, text_features)
        similarity = 100.0 * similarity
        similarity = similarity.softmax(dim=-1)
        # print(similarity[0])
        return self.head(X[:, 0]), similarity
    
    def training_step(self, batch):
        Y_hat, Sim = self(*batch[:-1])
        loss1 = self.loss(Y_hat, batch[-1])
        loss2 = self.loss(Sim, batch[-1])
        # lamd = 1 / self.num_classes
        lamd = 0.5
        loss = (1 - lamd) * loss1 + lamd * loss2
        # self.plot('loss1', loss1, train=True)
        # self.plot('loss2', loss2, train=True)
        self.plot('loss', loss, train=True)
        return loss
    
    def validation_step(self, batch):
        Y_hat, Sim = self(*batch[:-1])
        loss1 = self.loss(Y_hat, batch[-1])
        loss2 = self.loss(Sim, batch[-1])
        # lamd = 1 / self.num_classes
        lamd = 0.5
        loss = (1 - lamd) * loss1 + lamd * loss2
        acc = self.accuracy(Y_hat, batch[-1])
        # self.plot('loss1', loss1, train=False)
        # self.plot('loss2', loss2, train=False)
        self.plot('loss', loss, train=False)
        self.plot('acc', acc, train=False)
        return loss, acc
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)