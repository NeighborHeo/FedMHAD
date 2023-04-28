import torch
import numpy as np
import os

class EarlyStopper:
    def __init__(self, patience=5, delta=0, checkpoint_dir='checkpoints'):
        self.patience = patience
        self.delta = delta
        self.checkpoint_dir = checkpoint_dir
        
        self.best_loss = np.inf
        self.best_acc = 0
        self.counter = 0
        self.is_best = False
    
    def is_best_loss(self, score):
        if score < self.best_loss - self.delta:
            self.best_loss = score
            self.counter = 0
            self.is_best = True
        else:
            self.counter += 1
            self.is_best = False
        return self.is_best
    
    def is_best_accuracy(self, accuracy):
        if accuracy > self.best_acc + self.delta:
            self.best_acc = accuracy
            self.counter = 0
            self.is_best = True
        else:
            self.counter += 1
            self.is_best = False
        return self.is_best
    
    def save_checkpoint(self, model, epoch, loss, acc, filename=None):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        
        if self.is_best:
            if filename is None:
                filename = 'best_model.pth'
            print(f"Best checkpoint at epoch {epoch}, loss: {loss:.4f}, accuracy: {acc:.4f}")
            torch.save(model.state_dict(), os.path.join(self.checkpoint_dir, filename))
            
        if self.counter >= self.patience:
            print(f"Early stopping at epoch {epoch}, loss: {loss:.4f}, accuracy: {acc:.4f}")
            return True
        
        return False