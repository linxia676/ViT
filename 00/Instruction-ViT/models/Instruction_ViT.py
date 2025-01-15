import torch
from torch import nn
from timm.models.vision_transformer import VisionTransformer
from timm.models.layers import PatchEmbed
from timm.models.registry import register_model
import timm

class InstructionVisionTransformer(VisionTransformer):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        super().__init__(img_size,patch_size,in_chans, num_classes, embed_dim, depth,
                 num_heads, mlp_ratio, qkv_bias, representation_size, distilled,
                 drop_rate, attn_drop_rate, drop_path_rate, embed_layer, norm_layer,
                 act_layer, weight_init)
        self.prompt_proj = nn.Linear(512,embed_dim)
        
    def reset_prompt(self,prompt_token):
        self.prompt_token = nn.Parameter(torch.tensor(prompt_token,dtype=torch.float),requires_grad=True)
        self.prompt_num = self.prompt_token.shape[0]
    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        
        prompt_tokens = self.prompt_proj(self.prompt_token)
        x = torch.cat([x,prompt_tokens.expand(x.shape[0], -1, -1)],dim=1)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0]),x[:,-self.prompt_num:]
        else:
            return x[:, 0], x[:, 1],x[:,-self.prompt_num:]
        
    def forward_logit(self,image_features,prompt):
        image_features = torch.nn.functional.normalize(image_features,p=2,dim=1) 
        prompt = torch.nn.functional.normalize(prompt,p=2,dim=2)
        logits = torch.einsum('ik,ijk->ij', image_features, prompt)
        logits = logits*100
        logits = logits.softmax(dim=-1)
        return logits
    
    def forward(self, x):
        x, prompt = self.forward_features(x)
        raw_x = self.head(x)
        x = self.forward_logit(x,prompt)
        return x,raw_x
    
@register_model
def instruction_vit_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model = InstructionVisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    if pretrained:
        vit_model = timm.create_model('vit_base_patch16_224',pretrained=True, **kwargs)
        model.load_state_dict(vit_model.state_dict(),strict=False)
    return model