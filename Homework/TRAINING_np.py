from __future__ import print_function, division

import numpy as np 

class XOR(object):
    def __init__(self, **kwags):
        self.train = kwags.pop('train', True)
        self.lr = kwags.pop('learning_rate', 1)
        self.num_epochs = kwags.pop('num_epochs', 1000)
        self.checkpoint_name = kwags.pop('checkpoint', None)
        self.mse = []
    def _save_checkpoint(self):
        if self.checkpoint_name is None: return                                  
        checkpoint = {
            'leaning rate': self.lr,
            'mse': self.mse
            'weight': self.w1
        }
        filename = '%s_epoch_%d.pkl' % (self.checkpoint_name, self.epoch)
        if self.verbose:
            print('Saving checkpoint to "%s"' % filename)
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)

    def _foward(self):


