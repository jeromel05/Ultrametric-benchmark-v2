from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from torchmetrics.functional import accuracy, average_precision, auroc, roc, confusion_matrix
from torchmetrics.utilities.data import to_categorical
from util_functions import make_confusion_matrix_figure, make_roc_curves_figure, print_metrics

class FFNetwork(pl.LightningModule):
    def __init__(self, input_size, hidden_size, nb_classes, mode, optimizer, lr, lr_scheduler, 
                eval_steps, max_batches_per_epoch, b_len):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.l1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(self.hidden_size, nb_classes)
        self.softmax = nn.Softmax(dim = 1)

        self.run_val=True
        self.max_batches_per_epoch=max_batches_per_epoch

        self.save_hyperparameters()

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
            #factor = self.hparams.lr_factor
            #patience = self.hparams.lr_patience
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
        #print('train:', self.trainer.current_epoch, x.size(0))
        num_classes = y.size(1)
        x = x.view(x.size(0), -1)
        o = self.forward(x)
        loss = F.binary_cross_entropy(o, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        target = to_categorical(y, argmax_dim=1)
        train_acc = accuracy(o, target, average='micro', num_classes=num_classes)
        self.log('train_acc', train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        cond_um = self.hparams.mode == 'um'
        cond_um_no_shuffle = cond_um and (np.all(self.hparams.eval_steps == None))
        cond_um_shuffle = cond_um and (not np.all(self.hparams.eval_steps == None)) and (self.trainer.current_epoch in self.hparams.eval_steps)
        cond_last_batch = batch_idx == np.ceil(len(self.trainer.datamodule.train_dataloader().sampler.um_indexes) / self.trainer.datamodule.batch_size_train) - 1
        if cond_last_batch and (cond_um_no_shuffle or cond_um_shuffle):
            self.run_val=True
        return loss

    def validation_step(self, val_batch, batch_idx):
        if self.run_val==True:
            x, y = val_batch
            num_classes = y.size(1)
            x = x.view(x.size(0), -1)
            o = self.forward(x)
            loss = F.binary_cross_entropy(o, y)
            self.log('val_loss', loss) #, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            target = to_categorical(y, argmax_dim=1)
            
            val_acc, val_ap, val_auroc_, val_cf_mat, val_roc_curve = self.evaluate_metrics(o, 
                                                                target, num_classes, compute_cf_mat=False,
                                                                cm_figure=False, roc_figure=False)
            self.log('val_acc', val_acc) #, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            #self.log('val_ap', val_ap, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            #self.log('val_auroc', val_auroc_, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            if not np.all(self.hparams.eval_steps == None) and self.trainer.current_epoch in self.hparams.eval_steps:
                val_epoch = self.hparams.eval_steps.index(self.trainer.current_epoch)
            else:
                val_epoch = self.trainer.current_epoch
            self.log('val_epoch', float(val_epoch), on_step=True, on_epoch=False, prog_bar=False, logger=True)

            print(f"Val epoch: {self.trainer.current_epoch}, loss: {loss:.3}, acc: {val_acc:.3}")
            if not val_cf_mat == None:
                self.log_figures(num_classes, val_cf_mat, val_roc_curve)

            if (not np.all(self.hparams.eval_steps == None)) and (self.hparams.mode == 'um'):
                #print(f'UM RESET at epoch: {self.trainer.current_epoch}')
                self.reset_network_sampler()
        self.run_val=False
        
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        num_classes = y.size(1)
        x = x.view(x.size(0), -1)
        o = self.forward(x)
        loss = F.binary_cross_entropy(o, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        target = to_categorical(y, argmax_dim=1)
        test_acc, test_ap, test_auroc_, test_cf_mat, test_roc_curve = self.evaluate_metrics(o, 
                                                        target, num_classes, compute_cf_mat=True, 
                                                        cm_figure=True, roc_figure=True)

        self.log('test_acc', test_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('test_ap', test_ap, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('test_auroc', test_auroc_, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        #print_metrics(test_acc, test_ap, test_auroc_, test_cf_mat, test_roc_curve, save_figs = (num_classes <= 16))
        
    def validation_epoch_end(self, outputs):
        if self.hparams.mode == 'split':
            self.trainer.datamodule.train_dataloader().sampler.update_curr_epoch_nb()
            self.trainer.datamodule.val_dataloader().sampler.update_curr_epoch_nb()
        
    def log_figures(self, num_classes, val_cf_mat, val_roc_curve):
        tensorboard = self.logger.experiment
        if num_classes > 16:
            tensorboard.add_image("val_cf_mat", val_cf_mat, dataformats='HW', global_step=self.global_step)
        else:
            tensorboard.add_figure("val_cf_mat", val_cf_mat, global_step=self.global_step)
            tensorboard.add_figure("val_roc_curve", val_roc_curve, global_step=self.global_step)

    def reset_network_sampler(self): 
        self.trainer.datamodule.train_dataloader().sampler.reset_sampler()
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                #print(f"resetting layer: {layer}")
                layer.reset_parameters()

        if max(self.hparams.eval_steps) < self.trainer.current_epoch:
            print("Stopping because no more steps in eval steps list")
            self.trainer.should_stop = True    

    def on_train_epoch_start(self) -> None:
        self.run_val=False