from math import ceil
from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
import torchvision
from torchmetrics.functional import accuracy, average_precision, auroc, roc, confusion_matrix
from torchmetrics.utilities.data import to_categorical
from util_functions import bcolors
from util_functions import make_confusion_matrix_figure, make_roc_curves_figure

class FFNetwork(pl.LightningModule):
    def __init__(self, input_size, hidden_size, nb_classes, mode, optimizer, lr, lr_scheduler, 
                 b_len, eval_freq, eval_freq_factor, no_reshuffle, batch_size,  s_len, depth, 
                 keep_correlations, stoch_s_len, last_val_step=0, val_step=100):

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.l1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(self.hidden_size, nb_classes)
        self.softmax = nn.Softmax(dim = 1)

        self.criterion = self.loss_func()

        self.run_val=True
        self.last_val_step=last_val_step
        self.curr_val_step=last_val_step
        self.curr_reset_step=last_val_step
        self.init_eval_freq = eval_freq
        self.curr_eval_freq = eval_freq
        self.eval_freq_factor = eval_freq_factor
        self.last_val_acc = 0.0

        if mode == 'um':
            self.max_eval_freq = 15000
        if mode == 'split':
            self.max_eval_freq = 25000

        self.reset_network=False

        self.save_hyperparameters()
    
    def loss_func(self):
        return nn.BCELoss()

    ######
    def alt_forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def create_cnn_model(self):
        model = torchvision.models.resnet18(pretrained=False, num_classes=8)
        model.conv1 = nn.Conv2d(1, 28, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) # modify resnet for mnist ds
        model.maxpool = nn.Identity()
        return model
    ######

    def forward(self, x):
        hidden = self.l1(x)
        relu = self.relu(hidden)
        output = self.l2(relu)
        output = self.softmax(output)
        return output

    def configure_optimizers(self):
        if self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.2)
        
        if self.hparams.lr_scheduler == "reduce_lr":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=100, min_lr=1e-3)
            config_dict = {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss_step"}
            return config_dict
        else:
            return optimizer
    
    def evaluate_metrics(self, preds, target, num_classes, compute_cf_mat=False,
                         cm_figure=False, roc_figure=False):
        categorical_preds = to_categorical(preds, argmax_dim=1)

        acc = accuracy(preds, target, average='micro', num_classes=num_classes)
        ap = average_precision(preds, target, num_classes=num_classes, average='macro')
        auroc_ = auroc(preds, target, num_classes=num_classes, average='macro')
        cf_mat = None
        if compute_cf_mat:
            if (not cm_figure) or num_classes > 16:
                cf_mat = confusion_matrix(preds, target, num_classes=num_classes) #, normalize='true') # causes nan
            else:
                cf_mat = metrics.confusion_matrix(target.cpu().numpy(), categorical_preds.cpu().numpy(), labels=np.arange(num_classes))
                cf_mat = make_confusion_matrix_figure(cf_mat, np.arange(num_classes))
                #cf_mat = plot_to_image(cf_mat)
        
        roc_curve = roc(preds, target, num_classes=num_classes)
        fpr, tpr, _ = roc_curve
        if roc_figure and num_classes <= 16:
            roc_curve = make_roc_curves_figure(fpr, tpr, num_classes)
        
        return acc, ap, auroc_, cf_mat, roc_curve
  
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        #print('train:', self.trainer.current_epoch, x.size(), y.size())
        num_classes = y.size(1)
        x = x.view(x.size(0), -1)
        o = self.forward(x)
        loss = self.criterion(o, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        target = to_categorical(y, argmax_dim=1)
        #print('targets: ', target)
        train_acc = accuracy(o, target, average='micro', num_classes=num_classes)
        self.log('train_acc', train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        self.run_val=False
        if self.hparams.mode == 'rand': 
            length_epoch=len(self.trainer.datamodule.train_dataloader())
        else:
            length_epoch=len(self.trainer.datamodule.train_dataloader().sampler.um_indexes)

        last_batch_idx = length_epoch // self.trainer.datamodule.batch_size_train
        cond_last_batch = batch_idx == last_batch_idx
        cond_epoch_zero = (self.trainer.current_epoch==0 and batch_idx==0)

        if (((self.hparams.mode == 'rand') or (self.hparams.b_len == 0 and self.trainer.current_epoch % 5==0) or self.hparams.no_reshuffle) and cond_last_batch) or cond_epoch_zero:
            #print('batch_idx', batch_idx, last_batch_idx, len(y), list(set(target)), "aaa", target)
            self.run_val=True
        else:
            cond_b_len = self.trainer.global_step >= self.curr_eval_freq + self.last_val_step + self.curr_reset_step # cond to reset dataloader to 0
            cond_start_val = self.trainer.global_step > self.last_val_step + self.curr_reset_step and (self.trainer.global_step % self.hparams.val_step == 0) # cond to start validating at every val_step
            #print('cond_b_len', cond_b_len, 'cond_start_val', cond_start_val)
            
            if cond_start_val or cond_epoch_zero or cond_b_len:
                #print((self.trainer.current_epoch == 0), (not cond_um_shuffle), cond_um_shuffle, cond_b_len, cond_last_batch)
                self.run_val=True
            if cond_b_len:
                if self.hparams.b_len > 0: print(f'#steps taken for this eval: {self.trainer.global_step - self.last_val_step}')
                self.reset_network=True

        return loss

    def validation_step(self, val_batch, batch_idx):
        if self.run_val==True:
            x, y = val_batch
            num_classes = y.size(1)
            x = x.view(x.size(0), -1)
            o = self.forward(x)
            loss = self.criterion(o, y)
            self.log('val_loss', loss)
            target = to_categorical(y, argmax_dim=1)
            
            val_acc, _, _, val_cf_mat, val_roc_curve = self.evaluate_metrics(o, 
                                                                target, num_classes, compute_cf_mat=False,
                                                                cm_figure=False, roc_figure=False)
            
            self.log('val_acc', val_acc)
            #self.log('val_ap', val_ap)
            #self.log('val_auroc', val_auroc)
            if self.trainer.global_step > 0:
                self.set_curr_steps_epochs(is_reset=self.reset_network)
            self.log('val_step', float(self.curr_val_step))
            self.trainer.logger.save() # flushes to logger

            print(f"Val step: {self.curr_val_step}, tot_step: {self.trainer.global_step}, loss: {loss:.3}, acc: {val_acc:.3}")
            
            if not val_cf_mat == None:
                self.log_figures(num_classes, val_cf_mat, val_roc_curve)

            # Resets the network for a new run
            if self.reset_network and (not self.hparams.no_reshuffle) and self.hparams.mode in ['um', 'split'] and self.hparams.b_len > 0:
                until_idx=self.curr_val_step

                if self.trainer.current_epoch > 0:
                    self.adjust_eval_freq(val_acc)
                
                until_idx += self.curr_eval_freq
                print(f'{bcolors.OKGREEN}Chain SHUFFLED UNTIL: : {until_idx}, curr_eval_freq: {self.curr_eval_freq}{bcolors.ENDC}')
                self.reset_network_sampler(until_idx=until_idx)

            self.run_val=False
            self.reset_network=False
            self.last_val_acc = val_acc
        
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        num_classes = y.size(1)
        x = x.view(x.size(0), -1)
        o = self.forward(x)
        loss = self.criterion(o, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        target = to_categorical(y, argmax_dim=1)
        test_acc, test_ap, test_auroc_, test_cf_mat, test_roc_curve = self.evaluate_metrics(o, 
                                                        target, num_classes, compute_cf_mat=True, 
                                                        cm_figure=True, roc_figure=True)

        self.log('test_acc', test_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('test_ap', test_ap, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('test_auroc', test_auroc_, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
    def log_figures(self, num_classes, val_cf_mat, val_roc_curve):
        tensorboard = self.logger.experiment
        if num_classes > 16:
            tensorboard.add_image("val_cf_mat", val_cf_mat, dataformats='HW', global_step=self.global_step)
        else:
            tensorboard.add_figure("val_cf_mat", val_cf_mat, global_step=self.global_step)
            tensorboard.add_figure("val_roc_curve", val_roc_curve, global_step=self.global_step)

    def reset_network_sampler(self, until_idx=None): 
        self.trainer.datamodule.train_dataloader().sampler.reset_sampler(until_idx)
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def adjust_eval_freq(self, val_acc):
        if abs(self.last_val_acc - val_acc) < 0.05: self.eval_freq_factor *= 1.1
        elif abs(self.last_val_acc - val_acc) > 0.15: 
            self.eval_freq_factor *= 0.95
            if abs(self.last_val_acc - val_acc) > 0.25: self.eval_freq_factor *= 0.8
        if val_acc > 0.85: # stop exp growth at val_acc == 0.85
            self.eval_freq_factor = max(self.eval_freq_factor*0.6, 1.2)

        self.curr_eval_freq *= self.eval_freq_factor
        
        if self.curr_eval_freq > 10000:
            self.eval_freq_factor = 1.1
            if self.curr_eval_freq > self.max_eval_freq: # upper threshold for eval_freq, otherwise will never be reached and timeout
                self.curr_eval_freq = self.max_eval_freq
        if self.curr_eval_freq < max(self.hparams.b_len, self.init_eval_freq): # lower bound
            self.curr_eval_freq = max(self.hparams.b_len, self.init_eval_freq) * 2
            self.eval_freq_factor = 1.75
        if val_acc > 0.5 and self.curr_eval_freq < 1000:
                self.curr_eval_freq = 2000

        self.curr_eval_freq = int(ceil(self.curr_eval_freq / self.hparams.b_len) * self.hparams.b_len) # multiple of b_len

    def set_curr_steps_epochs(self, is_reset=False):
        self.curr_val_step = self.trainer.global_step
        if is_reset: 
            self.curr_reset_step = self.trainer.global_step

        if self.hparams.b_len > 0 and (not self.hparams.no_reshuffle):
            self.curr_val_step -= self.last_val_step
            if is_reset:                # only in the case of a reset are these params updates, bc they are in cond_b_len
                self.curr_reset_step -= self.last_val_step
                self.last_val_step = self.trainer.global_step
            assert(self.curr_val_step >= 0)

