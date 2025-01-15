from hao.ViT import ViT
from data_ISIC import ISICDataModule
import hao.utils as u
from d2l import torch as d2l

if __name__ == '__main__':
    img_size, patch_size, batch_size = 16, 8, 128
    num_workers = 4
    num_classes = 10
    max_epochs = 7
    num_gpus = 1
    use_bias = False
    num_hiddens, mlp_num_hiddens, num_heads, num_blks = 128, 512, 4, 2
    emb_dropout, blk_dropout, lr = 0.1, 0.1, 0.1
    last_model_path = r'model_pth/last_model.pth'
    best_model_path = r'model_pth/best_model.pth'

    data = d2l.FashionMNIST(batch_size, resize=(img_size, img_size))
    train_loader = data.get_dataloader(train=True)
    test_loader = data.get_dataloader(train=False)
    batch = next(iter(train_loader))
    data.visualize(batch, nrows=2, ncols=4)
    model = ViT(img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads,
                num_blks, emb_dropout, blk_dropout, lr, use_bias, num_classes)
    trainer = u.Trainer(max_epochs, num_gpus, last_model_path, best_model_path)
    trainer.fit(model, data)