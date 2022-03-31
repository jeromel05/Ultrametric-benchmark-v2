import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ProgressBarBase

class UltraMetricCallback(Callback):
    def __init__(self, eval_steps: list):
        super().__init__()
        self.eval_steps = eval_steps

    def on_train_epoch_end(self, trainer, pl_module):
        print("CALLBACK")
        print(trainer.current_epoch, self.eval_steps[0])

        if np.isclose(trainer.current_epoch, self.eval_steps[0], atol=10):
            trainer.validate(pl_module, datamodule=trainer.datamodule)

            print(f'UM RESET at epoch: {trainer.current_epoch}')
            trainer.datamodule.train_dataloader().sampler.reset_sampler()
            for layer in pl_module.children():
                if hasattr(layer, 'reset_parameters'):
                    #print(f"resetting layer: {layer}")
                    layer.reset_parameters()
            self.eval_steps = self.eval_steps[1:] #pop the first el bc we used it

class LitProgressBar(ProgressBarBase):
    def __init__(self):
        super().__init__()
    
    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items

    
            
