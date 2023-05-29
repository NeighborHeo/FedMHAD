import os
from poutyne import Model, ModelCheckpoint, CSVLogger
from poutyne.framework.callbacks.callbacks import Callback

class CometLogger(Callback):
    def __init__(self, experiment):
        self.experiment = experiment
        self.epoch = 0
    def on_epoch_end(self, epoch_number, logs):
        self.epoch = epoch_number
        self.experiment.log_metrics(logs, step=self.epoch)
    def on_valid_end(self, logs):
        self.experiment.log_metrics(logs, step=self.epoch)

def get_callbacks(save_path, experiment):
    print(f'Saving directory is {save_path}')
    
    if not os.path.exists(save_path):
        print(f'Creating directory {save_path}')
        os.makedirs(save_path)
        
    callbacks = [
        # Save the latest weights to be able to continue the optimization at the end for more epochs.
        ModelCheckpoint(os.path.join(save_path, 'last_weights.ckpt')),

        # Save the weights in a new file when the current model is better than all previous models.
        ModelCheckpoint(os.path.join(save_path, 'best_weight.ckpt'),
                        save_best_only=True, restore_best=True, verbose=True),

        # Save the losses for each epoch in a TSV.
        CSVLogger(os.path.join(save_path, 'log.tsv'), separator='\t'),
        
        CometLogger(experiment)
    ]
    
    return callbacks
