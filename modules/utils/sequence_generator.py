import numpy as np
import pickle
import random
import datetime
from tensorflow.keras.utils import Sequence


class SequenceGen(Sequence):
    def __init__(self, data_file, batch_size, drop_prob=0.1, emb_shape=65):
        self.FILLER = np.array(
            [-1] * emb_shape
        )  # n-dim embedding + 1 nb of days since prev scan indicator
        self.PAD_LEN = 10
        self.MX_DIFF = 3481  # maximum possible number of days between scans (for scaling)

        with open(data_file, 'rb') as handle:
            self.data = pickle.load(handle)

        self.eps = len(list(self.data.keys())) - 1
        self.drop_prob = drop_prob
        self.batch_size = batch_size
        self.pids = list(self.data[0].keys())
        self.num_samples = len(self.pids)
        self.shuffle = False

    def __len__(self):
        return int(np.floor(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        ep = random.randint(0, self.eps)
        batch = self.pids[self.batch_size * index:self.batch_size *
                          (index + 1)]
        x = []
        y = []

        for pid in batch:
            sample = self.data[ep][pid]
            samp, targ = self.format_sequence(sample)
            x.append(samp)
            y.append(targ)
            try:
                assert len(x) > 0 and len(y) > 0
            except AssertionError:
                import pdb
                pdb.set_trace()

        return np.array(x), np.array(y)

    def format_sequence(self, sample):
        out_seq = []
        prev_date = None
        targ = -1
        drop_countdown = np.max(len(sample.keys()) - 2, 0)
        dates = [
            datetime.datetime.strptime(d, '%m/%d/%Y') for d in sample.keys()
        ]
        dates = sorted(dates, reverse=True)
        for idx, date_obj in enumerate(dates):
            date = date_obj.strftime('%m/%d/%Y')
            tmp = random.randint(1, 100)

            if idx == 0:
                targ = sample[date][1]

            if drop_countdown > 0:
                if tmp < int(self.drop_prob * 100):
                    out_seq.append(self.FILLER)
                    drop_countdown -= 1
                else:
                    if prev_date:
                        t_diff = (date_obj - prev_date).days
                        t_diff /= self.MX_DIFF
                        seq = sample[date][0]
                        seq = np.append(seq, t_diff)
                    else:
                        seq = sample[date][0]
                        seq = np.append(seq, 0)
                    out_seq.append(seq)

                    prev_date = date_obj
            else:
                if prev_date:
                    t_diff = (date_obj - prev_date).days
                    t_diff /= self.MX_DIFF
                    seq = sample[date][0]
                    seq = np.append(seq, t_diff)
                else:
                    seq = sample[date][0]
                    seq = np.append(seq, 0)
                out_seq.append(seq)

                prev_date = date_obj

        while len(out_seq) < self.PAD_LEN:
            out_seq.append(self.FILLER)

        return np.array(out_seq), targ[0]