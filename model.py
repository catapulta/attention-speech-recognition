from torch import nn
from torch.nn.utils import rnn
import torch
import torch.nn.functional as F
import pdb
import numpy as np


class EncoderLSTM(nn.Module):
    def __init__(self, key_size=128, value_size=256, nlayers=3, hidden_size=256, bidirectional=True):
        super(EncoderLSTM, self).__init__()
        self.key_size = key_size
        self.value_size = value_size
        self.kv_size = key_size + value_size
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1
        self.rnn_input_size = 40
        self.hidden_size = hidden_size
        self.nlayers = nlayers

        self.rnn = []
        self.init_hidden = []
        rnn_input_size = self.rnn_input_size
        for i in range(nlayers):
            self.rnn.append(nn.GRU(input_size=rnn_input_size,
                                   hidden_size=self.hidden_size,
                                   num_layers=2,
                                   bidirectional=self.bidirectional,
                                   dropout=.1))
            self.init_hidden.append(nn.Parameter(torch.randn(self.directions *2, 1, hidden_size)))
            rnn_input_size = self.hidden_size*4
        self.init_hidden = nn.ParameterList(self.init_hidden)
        self.rnn = nn.ModuleList(self.rnn)

        # def key/value
        hidden_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        hidden_size = hidden_size * 2
        self.scoring = nn.Sequential(
            # nn.BatchNorm1d(hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, self.kv_size)
        )

        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, seq_list):
        batch_size = len(seq_list)
        mask = [len(s) for s in seq_list]  # lens of all inputs (sorted by loader)
        seq_list = rnn.pad_sequence(seq_list, batch_first=False)  # max_len, batch_size, features
        x = seq_list.cuda() if torch.cuda.is_available() else seq_list
        x = x[:2**self.nlayers * (x.shape[0] // (2**self.nlayers))]

        for i in range(self.nlayers):
            x, _ = self.rnn[i](x, self.init_hidden[i].expand(-1, batch_size, -1).contiguous())  # max_len/k, batch_size, features
            s = x.shape
            x = x.permute(1, 0, 2).contiguous().view(batch_size, s[0] // 2, s[2] * 2).permute(1, 0, 2)

        shrink_factor = 2 ** self.nlayers
        mask = [l // shrink_factor for l in mask]
        output_flatten = torch.cat(
            [x[:mask[i], i] for i in
             range(batch_size)])  # concatenated output (batch_size, sum(reduced_len), hidden)
        scores_flatten = self.scoring(output_flatten)  # concatenated scores (batch_size, sum(reduced_len), kv_size)
        cum_lens = np.cumsum([0] + mask)
        scores_unflatten = [scores_flatten[cum_lens[i]:cum_lens[i + 1]] for i in range(batch_size)]
        scores_unflatten = rnn.pad_sequence(scores_unflatten)  # max_len, batch, kv_size
        key = scores_unflatten[:, :, :self.key_size]  # max_len, batch, key_size
        value = scores_unflatten[:, :, self.key_size:]  # max_len, batch, value_size
        return key, value, mask  # return concatenated key, value, hidden state, mask


# Model that takes packed sequences in training
class EncoderCNN(nn.Module):
    def __init__(self, key_size=128, value_size=256, nlayers=3, hidden_size=512,
                 cnn_compression=2, bidirectional=True):
        super(EncoderCNN, self).__init__()
        self.cnn_compression = cnn_compression
        self.key_size = key_size
        self.value_size = value_size
        self.kv_size = key_size + value_size
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1
        self.rnn_input_size = 1280
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=(1, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 2), padding=(1, 1), bias=False),
            nn.ELU(),
            nn.Conv2d(64, self.rnn_input_size // 10, kernel_size=3, stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(self.rnn_input_size // 10),
            nn.ELU()
        )
        self.rnn = nn.GRU(input_size=self.rnn_input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.nlayers,
                          bidirectional=self.bidirectional,
                          dropout=.1)

        # define initial state
        self.init_hidden = torch.nn.Parameter(torch.randn(self.nlayers * self.directions, 1, self.hidden_size))

        # def key/value
        hidden_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        self.scoring = nn.Sequential(
            # nn.BatchNorm1d(hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, self.kv_size)
        )

        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, seq_list):
        '''
        :param seq_list: list of sequences ordered by size
        :return: scores (sum(seq_lens), kv_size)
        '''
        batch_size = len(seq_list)
        lens = [len(s) for s in seq_list]  # lens of all inputs (sorted by loader)
        seq_list = rnn.pad_sequence(seq_list, batch_first=True)  # batch_size, max_len, features
        seq_list = seq_list.cuda() if torch.cuda.is_available() else seq_list
        seq_list = seq_list.unsqueeze(1)  # create a channel for CNN: batch_size, 1, max_len, features
        embedding = self.cnn(seq_list)  # wasteful cnn computes over padded data: batch_size, 1, red_max_len, features
        n, c, h, w = embedding.size()  # h is (CNN reduced) max_len; w * c are features
        reduced_embeddings = []  # batch_size, reduced_len, features
        mask = []
        for i in range(batch_size):
            l = -(-lens[i] // self.cnn_compression)  # ceiling
            e = embedding[i, :, :l, :].permute(0, 2, 1).contiguous().view(c * w, l).permute(1,
                                                                                            0)  # reduced_len, features
            reduced_embeddings.append(e)
            mask.append(l)
        # reduced_embeddings = seq_list
        # mask = lens
        packed_input = rnn.pack_sequence(reduced_embeddings)  # packed uneven length sequences for fast RNN processing
        packed_input = packed_input.cuda() if torch.cuda.is_available() else packed_input
        # learn initial state
        init_hidden = self.init_hidden.expand(-1, batch_size, -1).contiguous()
        packed_output, hidden = self.rnn(packed_input, init_hidden)  # reduced_len, batch_size, features
        packed_output, _ = rnn.pad_packed_sequence(packed_output)  # unpacked output (padded)
        output_flatten = torch.cat(
            [packed_output[:mask[i], i] for i in
             range(batch_size)])  # concatenated output (batch_size, sum(reduced_len), hidden)
        scores_flatten = self.scoring(output_flatten)  # concatenated scores (batch_size, sum(reduced_len), kv_size)
        cum_lens = np.cumsum([0] + mask)
        scores_unflatten = [scores_flatten[cum_lens[i]:cum_lens[i + 1]] for i in range(batch_size)]
        scores_unflatten = rnn.pad_sequence(scores_unflatten)  # max_len, batch, kv_size
        key = scores_unflatten[:, :, :self.key_size]  # max_len, batch, key_size
        value = scores_unflatten[:, :, self.key_size:]  # max_len, batch, value_size
        return key, value, mask  # return concatenated key, value, mask


# Model that takes packed sequences in training
class DecoderRNN(nn.Module):
    def __init__(self, num_chars, key_size=128, value_size=256, nlayers=4,
                 hidden_size=512, teacher=0.4, bidirectional=False):
        super(DecoderRNN, self).__init__()
        self.teacher = teacher
        self.key_size = key_size
        self.value_size = value_size
        self.num_chars = num_chars
        self.bidirectional = bidirectional
        self.bidirectional = False
        self.directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.nlayers = nlayers

        # create RNN
        self.first_hidden = torch.nn.Parameter(torch.randn((1, self.hidden_size)))
        self.init_hidden = []
        self.cells = []
        for i in range(self.nlayers):
            if i == 0:
                input_size = self.num_chars
                hidden_size = self.hidden_size + self.value_size
            elif i == 1:
                input_size = self.hidden_size + self.value_size
                hidden_size = self.hidden_size
            else:
                input_size = hidden_size = self.hidden_size
            self.init_hidden.append(torch.nn.Parameter(torch.randn((1, hidden_size))))
            self.cells.append(torch.nn.GRUCell(input_size, hidden_size, bias=True))
        self.cells = nn.ModuleList(self.cells)
        self.init_hidden = nn.ParameterList(self.init_hidden)

        # create attention
        self.query = nn.Linear(self.hidden_size, self.key_size)
        self.query = self.query.cuda() if torch.cuda.is_available() else self.query
        self.plot_attention = []

        # create scoring
        hidden_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        self.scoring = nn.Sequential(
            # nn.BatchNorm1d(hidden_size),
            nn.Sigmoid(),
            nn.Linear(self.hidden_size, self.num_chars)
        )

        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, seq_list, keys, values, masks):
        assert len(seq_list) == keys.shape[1], 'batch sizes must match'
        assert len(masks) == keys.shape[1], 'batch sizes must match'
        assert values.shape[1] == keys.shape[1], 'batch sizes must match'

        # sort list according to target length
        batch_size = len(seq_list)
        lens = [len(s) for s in seq_list]  # lens of all inputs
        seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
        keys = torch.stack([keys[:, i] for i in seq_order], dim=1)
        values = torch.stack([values[:, i] for i in seq_order], dim=1)
        masks = [masks[i] for i in seq_order]
        seq_list = [seq_list[i] for i in seq_order]
        seq_list = rnn.pad_sequence(seq_list, batch_first=True)  # batch_size, max_len, features
        seq_list = seq_list.cuda() if torch.cuda.is_available() else seq_list

        hiddens = []
        matrix_mask = torch.zeros(keys.shape[1], 1, keys.shape[0])
        for i, mask in enumerate(masks):
            matrix_mask[i, 0, :mask] = 1
        matrix_mask = matrix_mask.cuda() if torch.cuda.is_available() else matrix_mask

        self.plot_attention = []

        rnn_pred = []
        for t in range(lens[0]):
            # teacher forcing
            if not (self.teacher > np.random.random() and t != 0):
                x = seq_list[:, t]
                x_onehot = torch.FloatTensor(batch_size, self.num_chars)
                x_onehot = x_onehot.cuda() if torch.cuda.is_available() else x_onehot
                x_onehot = x_onehot.zero_()
                x = x.long().unsqueeze(1) if len(x.size()) == 1 else x.long()
                x = x.cuda() if torch.cuda.is_available() else x
                x = x_onehot.scatter_(1, x, 1)
            else:
                x = gumbel_x

            if t == 0:
                last_hidden = self.init_hidden[-1]
                last_hidden = last_hidden.expand(batch_size, -1)
            else:
                last_hidden = hiddens[-1]
            query = self.query(last_hidden).unsqueeze(0)  # 1, batch_size, key_size

            # attention calculation
            # Your key and query needs to have the same dim as you multiply (1,128) Q with (128,T)
            # key for getting the energies for all timesteps. Once you have that (1,T), you simply
            # multiply them with your values which are (T,V) to get a context of (1,V). As mentioned
            # before "V" has no hard constraints.

            # query: 1, batch_size, hidden_size || keys: max_len, batch_size, key_size
            energy = torch.bmm(query.permute(1, 0, 2), keys.permute(1, 2, 0))  # batch_size, 1, max_len
            energy = energy * matrix_mask  # mask energy
            attention = F.softmax(energy, dim=2)  # along seq_len: batch_size, 1, max_len
            # attention = F.normalize(attention, p=1, dim=2)
            self.plot_attention.append(attention[0, 0])
            # context: values: max_len, batch_size, value_size
            context = torch.bmm(attention, values.permute(1, 0, 2))  # batch_size, 1, value_size
            context = context.permute(1, 0, 2)  # 1, batch_size, value_size

            first_hidden = torch.cat([context.squeeze(0), self.first_hidden.expand(batch_size, -1)], dim=1)
            for i, cell in enumerate(self.cells):
                if t == 0:
                    hiddens.append(cell(x, self.init_hidden[i].expand(batch_size, -1) if i > 0 else first_hidden))
                else:
                    hiddens[i] = cell(x, hiddens[i])
                x = hiddens[i]

                # teacher forcing
                if i == len(self.cells) - 1:
                    temp_out = self.scoring(x)
                    gumbel_x = F.gumbel_softmax(temp_out, hard=True)
            rnn_pred.append(x)

        self.plot_attention = torch.stack(self.plot_attention, dim=0)
        rnn_pred = torch.stack(rnn_pred)
        output_flatten = torch.cat(
            [rnn_pred[:lens[i], i] for i in
             range(batch_size)])  # concatenated output (sum(lens), hidden)
        scores_flatten = self.scoring(output_flatten)  # concatenated scores (sum(lens), num_chars)
        cum_lens = np.cumsum([0] + lens)
        scores_unflatten = [scores_flatten[cum_lens[i]:cum_lens[i + 1]] for i in range(batch_size)]
        scores_unflatten = rnn.pad_sequence(scores_unflatten, batch_first=True,
                                            padding_value=-99)  # batch, max_len, num_chars
        return scores_unflatten  # batch, max_len, num_chars


class LAS(nn.Module):
    def __init__(self, num_chars=33, key_size=128, value_size=256, encoder_depth=3, decoder_depth=4, encoder_hidden=512,
                 decoder_hidden=512, cnn_compression=2, enc_bidirectional=True, teacher=0.4):
        super(LAS, self).__init__()
        self.encoder = EncoderCNN(key_size=key_size, value_size=value_size, nlayers=encoder_depth,
                                  hidden_size=encoder_hidden, cnn_compression=cnn_compression,
                                  bidirectional=enc_bidirectional)
        self.decoder = DecoderRNN(num_chars=num_chars, key_size=key_size, value_size=value_size, nlayers=decoder_depth,
                                  hidden_size=decoder_hidden, teacher=teacher, bidirectional=False)

    def forward(self, input, target):
        enc_out = self.encoder(input)
        out = self.decoder(target, enc_out[0], enc_out[1], enc_out[2])
        return out  # batch, max_len, num_chars


class LAS2(nn.Module):
    def __init__(self, num_chars=33, key_size=128, value_size=256, encoder_depth=3, decoder_depth=5, encoder_hidden=256,
                 decoder_hidden=512, enc_bidirectional=True, teacher=0.4):
        super(LAS2, self).__init__()
        self.encoder = EncoderLSTM(key_size=key_size, value_size=value_size, nlayers=encoder_depth,
                                   hidden_size=encoder_hidden, bidirectional=enc_bidirectional)
        self.decoder = DecoderRNN(num_chars=num_chars, key_size=key_size, value_size=value_size, nlayers=decoder_depth,
                                  hidden_size=decoder_hidden, teacher=teacher, bidirectional=False)

    def forward(self, input, target):
        enc_out = self.encoder(input)
        out = self.decoder(target, enc_out[0], enc_out[1], enc_out[2])
        return out  # batch, max_len, num_chars


if __name__ == '__main__':
    import character_list

    enc = EncoderCNN(cnn_compression=2)
    dec = DecoderRNN(len(character_list.LETTERS) + 1)
    las = LAS2()

    # with torch.no_grad():
    # enc_out = enc([torch.ones((120, 40)), torch.ones((90, 40))])
    # print(enc_out[0].shape)
    # print(dec([torch.ones(5), torch.ones(2)], enc_out[0], enc_out[1], enc_out[2]).shape)
    # print(dec([torch.ones(3), torch.ones(2)], enc_out[0], enc_out[1], enc_out[2]))
    # print(las([torch.ones((120, 40)), torch.ones((90, 40))],
    #           [torch.ones(20), torch.ones(1)]).shape)

    # with torch.no_grad():
    #     enc_out = enc([torch.ones((120, 40)), torch.ones((90, 40))])
    # targets = [torch.ones(20), torch.ones(1)]
    # las.eval()
    batches = 64
    targets = [torch.ones(10)] #+ [torch.ones(9)] * (batches - 1)
    inputs = [torch.ones((120, 40))] #+ [torch.ones((90, 40))] * (batches - 1)
    # scores = las([torch.ones((120, 40)), torch.ones((90, 40))], targets)
    scores = las(inputs, targets)
    print(scores.shape)
    scores = scores.permute(0, 2, 1)  # batch_size, num_classes, seq_len

    input_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-99)

    optimizer = torch.optim.Adam(las.parameters(), lr=1e-3, weight_decay=1e-6)
    # optimizer = torch.optim.Adam(enc.parameters(), lr=1e-3, weight_decay=1e-6)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-99)
    criterion2 = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-99)
    idx = -1 if scores.shape[2] > 1 else None

    # targets = torch.cat((targets.long(), targets.long()), dim=1)
    print(scores[:, :, :idx].shape, targets[:, 1:].shape)
    loss = criterion(scores[:, :, :idx], targets[:, 1:].long())
    loss = loss
    print(loss)
    # loss.backward()
    # optimizer.step()
