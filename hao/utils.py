import torch
from torch import nn
import inspect
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
from tqdm.notebook import tqdm
import collections
from IPython import display
import os


class HyperParameters:
    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes."""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)

def show_images(imgs, num_rows, num_cols, titles=None, scale=2):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        try:
            img = img.detach().numpy()
        except:
            pass
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def use_svg_display():
    """Use the svg format to display a plot in Jupyter."""
    backend_inline.set_matplotlib_formats('svg')

def gpu(i=0):
    """Get a GPU device."""
    return torch.device(f'cuda:{i}')

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getstate__(self):
        # 定义如何序列化
        return {'x': self.x, 'y': self.y}

    def __setstate__(self, state):
        # 定义如何反序列化
        self.x = state['x']
        self.y = state['y']

class ProgressBoard(HyperParameters):
    """The board that plots data points in animation."""
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-*', '--o', '-.v', ':h', ',^', ':v', '--'], colors=['#155E95', '#5D8736', '#A888B5', '#C30E59', '#D2691E', '#FFD700', '#2E8B57'],
                 fig=None, axes=None, figsize=(8, 4.5), display=True):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        # Point = collections.namedtuple('Point', ['x', 'y'])
        if not hasattr(self, 'raw_points'):
            self.raw_points = collections.OrderedDict()
        if not hasattr(self, 'data'):
            self.data = collections.OrderedDict()
        if label not in self.raw_points:
            self.raw_points[label] = []
        if label not in self.data:
            self.data[label] = []
        points = self.raw_points[label]
        line = self.data[label]
        points.append(Point(x, y))
        if len(points) != every_n:
            return
        mean = lambda x: sum(x) / len(x)
        line.append(Point(mean([p.x for p in points]),
                          mean([p.y for p in points])))
        # print(f"Epoch {line[-1].x:<2}: {label} {line[-1].y:<4.4f}")
        points.clear()
        if not self.display:
            return
        use_svg_display()
        if self.fig is None:
            self.fig = plt.figure(figsize=self.figsize)
        plt_lines, labels = [], []
        for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
            plt_lines.append(plt.plot([p.x for p in v], [p.y for p in v], ls, color=color)[0])
            labels.append(k)
        axes = self.axes if self.axes else plt.gca()
        if self.xlim: axes.set_xlim(self.xlim)
        if self.ylim: axes.set_ylim(self.ylim)
        if not self.xlabel: self.xlabel = self.x
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)
        axes.legend(plt_lines, labels)
        plt.grid(True)
        display.display(self.fig)
        display.clear_output(wait=True)

    def draw_his(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-*', '--o', '-.v', ':h'], colors=['#155E95', '#5D8736', '#A888B5', '#C30E59'],
                 figsize=(8, 4.5), show_coords=False, coord_interval=3, plot_keys=None, fig=None, axes=None):
        if not xlabel: xlabel = self.xlabel
        if not ylabel: ylabel = self.ylabel
        if not xlim: xlim = self.xlim
        if not ylim: ylim = self.ylim
        plot_keys = plot_keys or list(self.data.keys())
        for key in plot_keys:
            if key not in self.data:
                print(f"Warning: Key '{key}' not found in data.")
                continue

        if fig is None:
            fig, axes = plt.subplots(figsize=figsize)
        plt_lines, labels = [], []
        for (k, v), ls, color in zip(self.data.items(), ls, colors):
            if k not in plot_keys: continue
            x, y = [p.x for p in v], [p.y for p in v]
            plt_lines.append(plt.plot(x, y, ls, color=color)[0])
            labels.append(k)

            if show_coords:
                for i, (x_i, y_i) in enumerate(zip(x, y)):
                    if i % coord_interval != 0: continue
                    axes.annotate(f'({x_i:.2f}, {y_i:.2f})', xy=(x_i, y_i),
                                textcoords="offset points", xytext=(5, 5),
                                ha='center', fontsize=8, color=color)

        axes = axes if axes else plt.gca()
        if xlim: axes.set_xlim(xlim)
        if ylim: axes.set_ylim(ylim)
        if xlabel: axes.set_xlabel(xlabel)
        if ylabel: axes.set_ylabel(ylabel)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        axes.legend(plt_lines, labels)
        plt.grid(True)
        if self.display: plt.show()

class Trainer(HyperParameters):
    """The base class for training models with data."""
    def __init__(self, max_epochs, last_model_path, best_model_path, restart_train=False, gpu_idx = 0, gradient_clip_val=0):
        self.save_hyperparameters()
        self.gpus = [gpu(i) for i in range(self.max_gpus())]

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def load_model(self):
        if os.path.exists(self.last_model_path) and not self.restart_train:
            checkpoint = torch.load(self.last_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch_his = checkpoint['epoch_his']
            self.model.board.data = checkpoint['board_data']
        else:
            self.epoch_his = {}
            self.epoch_his['train_losses'] = []
            self.epoch_his['val_losses'] = []
            self.epoch_his['val_acc'] = []
            self.epoch_his['best_loss'] = float('inf')
            self.epoch_his['last_epoch'] = -1

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = self.model.configure_optimizers()
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        self.load_model()

        self.strat_epoch = self.epoch_his['last_epoch'] + 1
        for self.epoch in range(self.strat_epoch, self.max_epochs):
            train_loss, val_loss, val_acc = self.fit_epoch()
            self.update_epoch_his(train_loss, val_loss, val_acc)

            if val_loss < self.epoch_his['best_loss']:
                self.epoch_his['best_loss'] = val_loss
                self.save_model(best_model = True)
            self.save_model()

    def update_epoch_his(self, train_loss, val_loss, val_acc):
        self.epoch_his['train_losses'].append(train_loss)
        self.epoch_his['val_losses'].append(val_loss)
        self.epoch_his['val_acc'].append(val_acc)
        self.epoch_his['last_epoch'] = self.epoch

    def max_gpus(self):
        self.num_gpus = torch.cuda.device_count()
        return self.num_gpus

    def save_model(self, best_model = False):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'epoch_his': self.epoch_his,
            'board_data': self.model.board.data
        }
        if best_model:
            # val_loss = self.epoch_his['val_losses'][-1]
            # val_acc = self.epoch_his['val_acc'][-1]
            # print(f'best loss {val_loss:<4.4f}, acc {100.0 * val_acc:<5.2f}%')
            torch.save(checkpoint, self.best_model_path)
        else:
            torch.save(checkpoint, self.last_model_path)

    def print_training_his(self, max_epochs):
        # best_train_loss, best_val_loss, best_val_acc= float('inf'),  float('inf'), 0
        for i in range(max_epochs):
            train_loss = self.epoch_his['train_losses'][i]
            val_loss = self.epoch_his['val_losses'][i]
            val_acc = self.epoch_his['val_acc'][i]
            # best_train_loss = min(best_train_loss, train_loss)
            print(f"Epoch {i + 1:<2}: train loss {train_loss:<4.4f} val loss {val_loss:<4.4f} val acc {100.0 * val_acc:<5.2f}%")
        best_epoch, best_val_loss = min(enumerate(self.epoch_his['val_losses']), key=lambda x: x[1])
        best_train_loss, best_val_acc = self.epoch_his['train_losses'][best_epoch], self.epoch_his['val_acc'][best_epoch]
        print(f"*Best epoch {best_epoch + 1:<2}: train loss {best_train_loss:<4.4f} val loss {best_val_loss:<4.4f} val acc {100.0 * best_val_acc:<5.2f}%")
        
        

    def fit_epoch(self):
        self.model.train()
        train_loss_sum, val_loss_sum, val_acc = 0, 0, 0
        self.train_batch_idx, self.val_batch_idx = 0, 0
        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch))
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:  # To be discussed later
                    self.clip_gradients(self.gradient_clip_val, self.model)
                self.optim.step()
                train_loss_sum += loss
            self.train_batch_idx += 1
        if self.val_dataloader is None:
            return
        self.model.eval()
        for batch in self.val_dataloader:
            with torch.no_grad():
                loss, acc = self.model.validation_step(self.prepare_batch(batch))
                val_loss_sum += loss
                val_acc += acc
            self.val_batch_idx += 1

        train_loss = train_loss_sum / self.num_train_batches
        val_loss = val_loss_sum / self.num_val_batches
        val_acc = val_acc / self.num_val_batches

        return train_loss, val_loss, val_acc

    def prepare_batch(self, batch):
        if self.gpus:
            batch = [e.to(self.gpus[self.gpu_idx]) for e in batch]
        return batch


    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        if self.gpus:
            model.to(self.gpus[0])
        self.model = model

    def clip_gradients(self, grad_clip_val, model):
        params = [p for p in model.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
        if norm > grad_clip_val:
            for param in params:
                param.grad[:] *= grad_clip_val / norm
