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
    def __init__(self, input_size, hidden_size, nb_classes, mode):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.l1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(self.hidden_size, nb_classes)
        self.softmax = nn.Softmax(dim = 1)

        self.mode = mode
        self.save_hyperparameters()

    def forward(self, x):
        hidden = self.l1(x)
        relu = self.relu(hidden)
        output = self.l2(relu)
        output = self.softmax(output)
        return output

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        elif self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.2)
        
        return optimizer
    
    def evaluate_metrics(self, preds, target, num_classes,
                         cm_figure=False, roc_figure=False):
        categorical_preds = to_categorical(preds, argmax_dim=1)

        acc = accuracy(preds, target, average='micro', num_classes=num_classes)
        ap = average_precision(preds, target, num_classes=num_classes, average='macro')
        auroc_ = auroc(preds, target, num_classes=num_classes, average='macro')
        if (not cm_figure) or num_classes > 16:
            cf_mat = confusion_matrix(preds, target, num_classes=num_classes) #, normalize='true') # causes nan
        else:
            cf_mat = metrics.confusion_matrix(target.cpu(), categorical_preds.cpu(), labels=np.arange(num_classes))
            cf_mat = make_confusion_matrix_figure(cf_mat, torch.arange(num_classes))
            #cf_mat = plot_to_image(cf_mat)
        
        roc_curve = roc(preds, target, num_classes=num_classes)
        fpr, tpr, _ = roc_curve
        if roc_figure and num_classes <= 16:
            roc_curve = make_roc_curves_figure(fpr, tpr, num_classes)
        
        return acc, ap, auroc_, cf_mat, roc_curve
        
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        num_classes = y.size(1)
        x = x.view(x.size(0), -1)
        o = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(o, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        target = to_categorical(y, argmax_dim=1)
        
        roc_figure = True if (self.global_step % 100 == 0) and (num_classes <= 16) else False
        train_acc, train_ap, train_auroc, train_cf_mat, train_roc = self.evaluate_metrics(o, target, num_classes, 
                                                                            cm_figure=False, roc_figure=roc_figure)
        
        self.log('train_acc', train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_ap', train_ap, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_auroc', train_auroc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        tensorboard = self.logger.experiment
        tensorboard.add_image("train_cf_mat", train_cf_mat, dataformats='HW', global_step=self.global_step)
        #tensorboard.add_figure("train_cf_mat", train_cf_mat)
        
        if num_classes <= 16 and roc_figure:
            tensorboard.add_figure("train_roc_curve", train_roc, global_step=self.global_step)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        num_classes = y.size(1)
        x = x.view(x.size(0), -1)
        o = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(o, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        target = to_categorical(y, argmax_dim=1)
        
        val_acc, val_ap, val_auroc_, val_cf_mat, val_roc_curve = self.evaluate_metrics(o, 
                                                                    target, num_classes, cm_figure=True, roc_figure=True)

        self.log('val_acc', val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_ap', val_ap, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_auroc', val_auroc_, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        tensorboard = self.logger.experiment
        if num_classes > 16:
            tensorboard.add_image("val_cf_mat", val_cf_mat, dataformats='HW', global_step=self.global_step)
        else:
            tensorboard.add_figure("val_cf_mat", val_cf_mat, global_step=self.global_step)
            tensorboard.add_figure("val_roc_curve", val_roc_curve, global_step=self.global_step)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        num_classes = y.size(1)
        x = x.view(x.size(0), -1)
        o = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(o, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        target = to_categorical(y, argmax_dim=1)
        test_acc, test_ap, test_auroc_, test_cf_mat, test_roc_curve = self.evaluate_metrics(o, 
                                                        target, num_classes, cm_figure=True, roc_figure=True)

        self.log('test_acc', test_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_ap', test_ap, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_auroc', test_auroc_, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        print_metrics(test_acc, test_ap, test_auroc_, test_cf_mat, test_roc_curve, save_figs = (num_classes <= 16))
        
        
    def validation_epoch_end(self, outputs):
        if self.mode == 'split':
            self.trainer.datamodule.train_dataloader().sampler.update_curr_epoch_nb()
            self.trainer.datamodule.val_dataloader().sampler.update_curr_epoch_nb()

        if self.mode == 'um': # and (not self.current_epoch % 3):
            self.trainer.datamodule.train_dataloader().sampler.reset_sampler()
            for layer in self.children():
                if hasattr(layer, 'reset_parameters'):
                    #print(f"resetting layer: {layer}")
                    layer.reset_parameters()
