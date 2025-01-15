import clip
import numpy as np
import torch
import torch.nn.functional as F

device='cuda:5'
clipmodel, preprocess = clip.load("ViT-B/16", device=device)

name_list_ = [
            'Atelectasis',
            'Cardiomegaly',
            'Consolidation',
            'Edema',
            'Enlarged Cardiomediastinum',
            'Fracture',
            'Lung Lesion',
            'Lung Opacity',
            'No Finding',
            'Pleural Effusion',
            'Pleural Other',
            'Pneumonia',
            'Pneumothorax',
            'Support Devices']
            
print(name_list_)
text_prompts = []
for classname in name_list_:
    texts = ['A chest X-ray with '+classname+' symptoms.']
    texts = clip.tokenize(texts).to(device)  # tokenize
    class_embeddings = clipmodel.encode_text(texts)
    class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0).detach().cpu()
    text_prompts.append(class_embedding)
text_prompts = torch.stack(text_prompts, dim=1)
text_prompts = text_prompts.transpose(1,0)
np.save('text_feature.npy',text_prompts)