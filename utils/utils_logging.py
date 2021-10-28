import logging
import os
import sys


class AverageMeter(object):
    """Computes and stores the average and current value
    """

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_logging(log_root, rank, models_root,log_name='training.log'):
    log_root.setLevel(logging.DEBUG)
    if rank == 0:
        handler_stream = logging.StreamHandler(stream=sys.stdout)
        handler_stream.setLevel(logging.INFO)
        formatter = logging.Formatter("Training: %(asctime)s-%(message)s")
        handler_stream.setFormatter(formatter)
        log_root.addHandler(handler_stream)

        handler_file = logging.FileHandler(os.path.join(models_root, log_name),mode='w')
        handler_file.setLevel(logging.DEBUG)
        handler_file.setFormatter(formatter)
        log_root.addHandler(handler_file)
        log_root.info('rank_id: %d' % rank)
