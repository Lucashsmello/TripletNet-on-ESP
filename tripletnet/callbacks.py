import skorch
from skorch.callbacks import Callback


class LoadEndState(Callback):
    def __init__(self, checkpoint: skorch.callbacks.Checkpoint):
        self.checkpoint = checkpoint

    def on_train_end(self, net,
                     X=None, y=None, **kwargs):
        net.load_params(checkpoint=self.checkpoint)
