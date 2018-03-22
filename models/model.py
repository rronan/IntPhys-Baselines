import torch
import os

class Model(object):
    def __init__(self, opt):
        __name__ = 'model'

    def step(self, batch):
        # process one optimization step, from a batch
        pass

    def load(self, d):
        # must be rewritten if self is not nn.Module
        name = d[0][0] if len(d[0]) > 1 else 'model'
        path = d[0][-1]
        print('loading %s: %s' %(name, path))
        self.load_state_dict(torch.load(path))

    def save(self, path, epoch):
        # must be rewritten if self is not nn.Module
        f = open(os.path.join(path, '%s.txt' %self.__name__), 'w')
        f.write(str(self))
        f.close()
        torch.save(
            self.state_dict(),
            os.path.join(path, '%s_%s.pth' %(self.__name__, epoch))
        )

    def output():
        raise NotImplementedError

    def gpu(self):
        # must be rewritten if self is not nn.Module
        self.cuda()

    def score():
        raise NotImplementedError
