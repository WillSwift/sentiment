import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Model(nn.Module):
    def __init__(self, model_type='lstm', dim=768, nlayers=2,
                 bidir=True, dropout=0.0):
        super().__init__()
        self.rnn = EncoderRNN(model_type, dim, nlayers, bidir, dropout)
        self.lin = nn.Linear(dim, 3)

    def forward(self, input, input_len):
        '''
        input      : B x len x d,
        input_len  : B,
        '''
        feature = self.rnn(input, input_len)   # B x len x d
        feature = feature.mean(dim=1)
        logit = self.lin(feature)

        return logit 
 

class EncoderRNN(nn.Module):
    def __init__(self, model_type, num_units, nlayers, bidir, dropout):
        super().__init__()
        if model_type == 'lstm':
            self.rnn = nn.LSTM(num_units, num_units//2 if bidir else num_units,
                               nlayers, batch_first=True, bidirectional=bidir,
                               dropout=dropout)
        elif model_type == 'gru':
            self.rnn = nn.GRU(num_units, num_units//2 if bidir else num_units,
                              nlayers, batch_first=True, bidirectional=bidir,
                              dropout=dropout)
        elif model_type == 'affine':
            self.linear = nn.Linear(num_units, num_units)
        else:
            raise NotImplementedError

    def forward(self, input, input_len=None, return_last=False):
        if getattr(self, 'rnn', None) is None:   # affine
            return self.linear(input)
        if input_len is None:
            output, _ = self.rnn(input)
            return output
        packed_input = pack_padded_sequence(input, input_len, batch_first=True,
                                            enforce_sorted=False)
        packed_output, (hidden, _) = self.rnn(packed_input)
        if return_last:
            return hidden.permute(1, 0, 2).contiguous().view(input.size(0), -1)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        return output


if __name__ == '__main__':
    bsz, seq_len, dim = 4, 10, 20
    model = Model(dim=dim).to('cuda')
    input = torch.Tensor(bsz, seq_len, dim).normal_().to('cuda')
    input_len = [7, 10, 5, 9]
    y = model(input, input_len)
    
