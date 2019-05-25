#-*-: coding: utf-8

import os, sys, random
import ujson as json
import glob
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM



class DataLoader(object):
    def __init__(self, train_root, test_root):
        self.train_set = self._read_root_folder(train_root)
        random.shuffle(self.train_set)

        # TODO temporary..
        #self.test_set = self._read_root_folder(test_root)
        self.train_set = self.train_set[:-100]
        self.test_set = self.train_set[-100:]

        # iter
        self._iter = {'train': 0, 'test': 0}

        # BERT init
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.model = BertModel.from_pretrained('bert-base-chinese').to('cuda')
        self.model.eval()

        # Bertify everything
        self.train_set_bert = []
        for i, (x, _) in enumerate(self.train_set):
            self.train_set_bert.append(self._bertify(x).to('cpu'))
        self.train_set = list(zip(self.train_set, self.train_set_bert))
        self.test_set_bert = []
        for i, (x, _) in enumerate(self.test_set):
            self.test_set_bert.append(self._bertify(x).to('cpu'))
        self.test_set = list(zip(self.test_set, self.test_set_bert))



    def _read_root_folder(self, path):
        if not os.path.exists(path):
            print(f"Cannot find {path}", file=sys.stderr)
            sys.exit(-1)
        content = []
        for id in ['positive', 'neutral', 'negative']:
            content.extend(self._read_folder(os.path.join(path, id)))
        return content

    def _read_folder(self, path):
        if not os.path.exists(path):
            print(f"Cannot find {path}", file=sys.stderr)
            sys.exit(-1)
        content = []
        for fid in glob.glob(f"{path}/*baidu.txt"):
            _content = self._read_txt(fid)
            content.extend(_content)
        return content

    def _read_txt(self, path):
        with open(path, 'r') as f:
            content = json.load(f)
        content = list(map(lambda x: (x['text'], x['items'][0]['sentiment']),
                           content))
        return content

    def _reset_iter(self, set):
        self._iter[set] = 0

    def _bertify(self, x):
        x = self.tokenizer.tokenize(x)
        if len(x) > 512:
            x = x[:512]
        x = self.tokenizer.convert_tokens_to_ids(x)
        x = torch.tensor([x]).to('cuda')
        with torch.no_grad():
            encoded_layers, _ = self.model(x)
        return encoded_layers[-1]

    def get_random_batch(self, batch_size, set='train'):
        x, y = [], []
        this_set = self.train_set if set == 'train' else self.test_set
        for _ in range(batch_size):
            (__, _y), _x = random.choice(this_set)
            x.append(_x)
            y.append(_y)
        max_len = max([_x.size(1) for _x in x])
        xx = torch.Tensor(batch_size, max_len, x[-1].size(-1)).zero_()
        for b in range(batch_size):
            xx[b][:x[b].size(1)].copy_(x[b].squeeze(0))
        yy = torch.tensor(y)
        return xx, yy

    def get_iter_batch(self, batch_size, set='train'):
        x, y = [], []
        this_set = self.train_set if set == 'train' else self.test_set
        for b in range(batch_size):
            try:
                (__, _y), _x = this_set[self._iter[set] + b]
                self._iter[set] += 1
            except IndexError:
                self._reset_iter(set)
                break
            x.append(_x)
            y.append(_y)
        max_len = max([_x.size(1) for _x in x])
        xx = torch.Tensor(len(x), max_len, x[-1].size(-1)).zero_()
        for b in range(len(x)):
            xx[b][:x[b].size(1)].copy_(x[b].squeeze(0))
        yy = torch.tensor(y)
        return xx, yy


if __name__ == '__main__':
    random.seed(1)
    dl = DataLoader('/tmp/haitong_ai/sentiment/baidu_sentiment',
                    '/tmp/haitong_ai/sentiment/baidu_sentiment')
    dl.get_random_batch(32, 'train')
    while True:
        x, y = dl.get_iter_batch(32, 'test')
        print(x.size())
    

