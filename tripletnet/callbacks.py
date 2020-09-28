import skorch
from skorch.callbacks import Callback
from math import log10


class LoadEndState(Callback):
    def __init__(self, checkpoint: skorch.callbacks.Checkpoint):
        self.checkpoint = checkpoint

    def on_train_end(self, net,
                     X=None, y=None, **kwargs):
        net.load_params(checkpoint=self.checkpoint)


class LRMonitor(Callback):
    """
    Monitors the learning rate
    """

    def on_epoch_end(self, net, **kwargs):
        for group in net.optimizer_.param_groups:
            net.history.record('log10(lr)', log10(group['lr']))
            break



class CleanNetCallback(Callback):
    def on_train_end(self, net, X=None, y=None, **kwargs):
        net.callbacks = None
        net.callbacks_ = None
        net.history_ = None
        # net.virtual_params_ = None
        # net.criterion_ = None
        # net.optimizer_ = None
