from matplotlib import pyplot as plt
from sklearn import metrics
import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from torchmetrics.functional import accuracy, average_precision, auroc, roc, confusion_matrix
from torchmetrics.utilities.data import to_categorical
from repo.code.utils.util_functions import make_confusion_matrix_figure, make_roc_curves_figure

from util_functions import plot_confusion_matrix

class FFNetwork(pl.LightningModule):
    def __init__(self, input_size, hidden_size, learning_rate, nb_classes, mode):
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
    
    def evaluate_metrics(self, preds, target, num_classes,
                         cm_figure=False, roc_figure=False):
        categorical_preds = to_categorical(preds, argmax_dim=1)

        acc = accuracy(preds, target, average='micro', num_classes=num_classes)
        ap = average_precision(preds, target, num_classes=num_classes, average='macro')
        auroc_ = auroc(preds, target, num_classes=num_classes, average='macro')
        if (not cm_figure) or num_classes > 8:
            cf_mat = confusion_matrix(preds, target, num_classes=num_classes) #, normalize='true') # causes nan
        else:
            cf_mat = metrics.confusion_matrix(target, categorical_preds, labels=self.classes)
            cf_mat = make_confusion_matrix_figure(cf_mat, self.classes)
            #cf_mat = plot_to_image(cf_mat)
        
        roc_curve = roc(preds, target, num_classes=num_classes)
        fpr, tpr, _ = roc_curve
        if roc_figure and num_classes <= 8:
            roc_curve = make_roc_curves_figure(fpr, tpr, num_classes)
        
        return acc, ap, auroc_, cf_mat, roc_curve
        
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        num_classes = y.size(1)
        x = x.view(x.size(0), -1)
        o = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(o, y)
        self.log('train_loss', loss)
        
        target = to_categorical(y, argmax_dim=1)
        
        roc_figure = True if self.global_step % 100 == 0 else False
        train_acc, train_ap, train_auroc, train_cf_mat, train_roc = self.evaluate_metrics(o, target, num_classes, 
                                                                            cm_figure=False, roc_figure=roc_figure)
        
        self.log('train_acc', train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_ap', train_ap, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_auroc', train_auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        tensorboard = self.logger.experiment
        tensorboard.add_image("train_cf_mat", train_cf_mat, dataformats='HW', global_step=self.global_step)
        #tensorboard.add_figure("train_cf_mat", train_cf_mat)
        
        if num_classes <= 8:
            tensorboard.add_figure("train_roc_curve", train_roc, global_step=self.global_step)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        num_classes = y.size(1)
        x = x.view(x.size(0), -1)
        o = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(o, y)
        self.log('val_loss', loss)
        
        target = to_categorical(y, argmax_dim=1)
        
        val_acc, val_ap, val_auroc_, val_cf_mat, val_roc_curve = self.evaluate_metrics(o, 
                                                                    target, num_classes, cm_figure=True, roc_figure=True)

        self.log('val_acc', val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_ap', val_ap, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_auroc', val_auroc_, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        tensorboard = self.logger.experiment
        if num_classes > 8:
            tensorboard.add_image("val_cf_mat", val_cf_mat, dataformats='HW', global_step=self.global_step)
        else:
            tensorboard.add_figure("val_cf_mat", val_cf_mat, global_step=self.global_step)
            tensorboard.add_figure("val_roc_curve", val_roc_curve, global_step=self.global_step)
        
    def on_validation_end(self):
        if self.mode == 'split':
            self.trainer.datamodule.train_dataloader().sampler.update_curr_epoch_nb()
            self.trainer.datamodule.val_dataloader().sampler.update_curr_epoch_nb()
