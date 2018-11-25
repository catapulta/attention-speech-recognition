import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import character_list
# import Levenshtein as L
import logging
from model import LAS
import pdb

torch.multiprocessing.set_sharing_strategy('file_system')
logging.basicConfig(filename='train.log', level=logging.DEBUG)


# data loader
class UtteranceDataset(Dataset):
    def __init__(self, data_path='./data/dev.npy', label_path='./data/dev_transcripts.npy', test=False):
        self.letter_dict = {j: i + 1 for i, j in enumerate(character_list.LETTERS)}
        self.letter_dict['<>'] = 0
        self.test = test
        self.data = np.load(data_path, encoding='latin1')
        labels = np.load(label_path) if not test else None  # index labels from 1 to n_labels
        if labels is not None:
            self.labels = []
            for words in labels:
                words = np.array([0]
                                 + [self.letter_dict[letter] for letter in ' '.join(words.astype(str)) if letter != '_']
                                 + [0])
                self.labels.append(words)
            self.labels = np.array(self.labels)
        self.num_entries = len(self.data)
        self.num_entries = int(len(self.data)*.001) if not 'test' in data_path else int(len(self.data)*.1)

    def __getitem__(self, i):
        data = self.data[i]
        data = torch.from_numpy(data)
        if self.test:
            return data
        else:
            labels = self.labels[i]
            labels = torch.from_numpy(labels)
            return data, labels

    def __len__(self):
        return self.num_entries


def collate(batch):
    '''
    Collate function. Transform a list of different length sequences into a batch. Passed as an argument to the DataLoader.
    seq_list: list with size batch_size. Each element is a tuple where the first element is the predictor
    data and the second element is the label.
    output: data on format (batch_size, var_len_sequence)
    '''
    if len(batch[0]) == 2:
        utts, labels = zip(*batch)
        lens = [seq.size(0) for seq in utts]
        seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
        utts = [utts[i] for i in seq_order]
        labels = [labels[i] for i in seq_order]
        return utts, labels
    else:
        utts = batch
        lens = [seq.size(0) for seq in utts]
        seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
        utts = [utts[i] for i in seq_order]
        return utts


class Levenshtein:
    def __init__(self, charmap):
        self.label_map = ['<>'] + charmap  # add special char to first entry

    def __call__(self, prediction, target):
        return self.forward(prediction, target)

    def forward(self, prediction, target):
        ls = 0.
        for i in range(len(target)):
            pred = "".join(self.label_map[j-1] for j in prediction[i].numpy())
            true = "".join(self.label_map[j-1] for j in target[i].numpy())
            ls += L.distance(pred, true)
        return ls


# model trainer
class LanguageModelTrainer:
    def __init__(self, model, loader, val_loader, test_loader, max_epochs=1, chars=character_list.LETTERS):
        self.model = model.cuda() if torch.cuda.is_available() else model
        self.chars = chars
        self.loader = loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_losses = []
        self.val_losses = []
        self.predictions = []
        self.predictions_test = []
        self.generated_logits = []
        self.generated = []
        self.generated_logits_test = []
        self.generated_test = []
        self.epochs = 0
        self.max_epochs = max_epochs

        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
        # self.optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, weight_decay=1e-6, momentum=0.9)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean', ignore_index=-99)
        self.criterion = self.criterion.cuda() if torch.cuda.is_available() else self.criterion
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=2)
        self.LD = Levenshtein(self.chars)
        self.best_rate = 1e10

    def train(self):
        self.model.train()  # set to training mode
        for epoch in range(self.max_epochs):
            epoch_loss = 0
            training_epoch_loss = 0
            for batch_num, (inputs, targets) in enumerate(self.loader):
                # # debug
                # # Save init values
                # old_state_dict = {}
                # for key in model.state_dict():
                #     old_state_dict[key] = model.state_dict()[key].clone()
                #
                # # Your training procedure
                # loss = self.train_batch(inputs, targets)
                #
                # # Save new params
                # new_state_dict = {}
                # for key in model.state_dict():
                #     new_state_dict[key] = model.state_dict()[key].clone()
                #
                # # Compare params
                # for key in old_state_dict:
                #     if (old_state_dict[key] == new_state_dict[key]).all():
                #         print('No diff in {}'.format(key))
                # print('Batch loss is ', float(loss))

                targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
                targets = targets.cuda() if torch.cuda.is_available() else targets
                inputs = inputs.cuda() if torch.cuda.is_available() else inputs

                loss = self.train_batch(inputs, targets)
                epoch_loss += loss
                training_epoch_loss += loss
                # training print
                batch_print = 40
                if batch_num % batch_print == 0 and batch_num != 0:
                    self.print_training(batch_num, self.loader.batch_size, training_epoch_loss, batch_print)
                    training_epoch_loss = 0

            epoch_loss = epoch_loss / (batch_num + 1)
            self.epochs += 1
            self.scheduler.step(epoch_loss)
            print('[TRAIN]  Epoch [%d/%d]   Loss: %.4f'
                  % (self.epochs, self.max_epochs, epoch_loss))
            self.train_losses.append(epoch_loss)
            # log loss
            tLog.log_scalar('training_loss', epoch_loss, self.epochs)
            # log values and gradients of parameters (histogram summary)
            for tag, value in self.model.named_parameters():
                tag = tag.replace('.', '/')
                tLog.log_histogram(tag, value.data.cpu().numpy(), self.epochs)
                tLog.log_histogram(tag + '/grad', value.grad.data.cpu().numpy(), self.epochs)
            # save
            torch.save(self.model.state_dict(), "models/{}.pt".format(epoch))

            # every 1 epochs, print validation statistics
            epochs_print = 1
            if self.epochs % epochs_print == 0 and not self.epochs == 0:
                with torch.no_grad():
                    self.model.eval()
                    t = "#########  Epoch {} #########".format(self.epochs)
                    print(t)
                    logging.info(t)
                    ls = 0
                    lens = 0
                    for j, (val_inputs, val_labels) in (enumerate(self.val_loader)):
                        idx = np.random.randint(0, len(val_inputs))
                        print('Ground', ''.join([self.chars[j-1] for j in val_labels[idx]]))
                        val_output, feature_lengths = self.gen_greedy_search(val_inputs, 190)
                        print('Pred', ''.join([self.chars[j-1] for j in val_output[idx:idx + 1]]))
                        ls += self.LD.forward(val_output, val_labels)
                        lens += len(val_inputs)
                    ls /= lens
                    t = "Validation LD {}:".format(ls)
                    print(t)
                    logging.info(t)
                    t = '--------------------------------------------'
                    print(t)
                    logging.info(t)
                    # log loss
                    vLog.log_scalar('LD', ls, self.epochs)
                    if self.best_rate > ls:
                        torch.save(self.model.state_dict(), "models/best.pt")
                        self.best_rate = ls
                    self.model.train()

    def print_training(self, batch_num, batch_size, loss, batch_print):
        t = 'At {:.0f}% of epoch {}'.format(
            batch_num * batch_size / self.loader.dataset.num_entries * 100, self.epochs)
        print(t)
        logging.info(t)
        t = "Training perplexity: {}".format(np.exp(loss / batch_print))
        print(t)
        logging.info(t)
        t = '--------------------------------------------'
        print(t)
        logging.info(t)

    def train_batch(self, inputs, targets):
        print('input size', (len(inputs), len(inputs[0])))
        scores = self.model(inputs, targets)
        scores = scores.permute(0, 2, 1)  # batch_size, num_classes, seq_len
        loss = self.criterion(scores, targets[:, 1:])
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return float(loss)  # avoid autograd retention

    def test(self):
        preds = []
        for i, inputs in enumerate(self.test_loader):
            pred = self.gen_random_search(inputs, 100, 190)
            pred = [self.chars[j - 1] for j in pred]
            preds.append(pred)
        return preds

    def gen_greedy_search(self, data_batch, max_len):
        prediction = []  # store predictions
        enc_out = self.model.encoder(data_batch)
        starts = torch.zeros(1, len(self.chars)+1)
        prediction.append(starts)
        scores = self.model.decoder(starts, enc_out[0], enc_out[1], enc_out[3])
        for i in range(max_len-1):
            words = torch.argmax(scores, dim=1).float()
            prediction.append(words)
            scores = self.model.decoder(words, enc_out[0], enc_out[1], enc_out[3])
        prediction = torch.stack(prediction, dim=1).squeeze(0)

        # remove excess words
        lens = torch.argmin(prediction, dim=1).long().tolist()  # finds the 0s in the prediction
        assert len(lens) == len(prediction), 'lens and prediction dont match'
        prediction = [prediction[i, :lens[i]+1] for i in range(len(prediction))]
        seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
        prediction = [prediction[i] for i in seq_order]
        return prediction, lens

    def gen_random_search(self, data_batch, random_paths, max_len):
        loss = torch.nn.CrossEntropyLoss(reduction='sum')

        enc_out = self.model.decoder(data_batch)
        starts = torch.zeros(1, len(self.chars)+1)
        prediction = []  # store predictions
        losses = []
        for iter in range(random_paths):
            rand_pred = list()  # store batch_preds
            rand_pred.append(starts)
            scores = self.model.decoder(starts, enc_out[0], enc_out[1], enc_out[3])
            for i in range(max_len-1):
                scores = F.softmax(scores, dim=1)
                words = torch.multinomial(scores, 1, replacement=False)
                rand_pred.append(words)
                scores = self.model.decoder(words, enc_out[0], enc_out[1], enc_out[3])
            rand_pred = torch.stack(rand_pred, dim=1)
            lens = torch.argmin(rand_pred, dim=1).tolist()  # finds the 0s in the prediction
            assert len(lens) == len(rand_pred), 'lens and prediction dont match'
            rand_pred = [rand_pred[i, :lens[i]] for i in range(len(rand_pred))]
            seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
            rand_pred = [rand_pred[i] for i in seq_order]
            scores = self.model.decoder(rand_pred, enc_out[0], enc_out[1], enc_out[3])
            prediction.append(rand_pred)
            rand_pred = torch.nn.utils.rnn.pad_sequence(rand_pred, batch_first=True, padding_value=-99)
            rand_pred = rand_pred.permute(0, 2, 1)  # batch_size, num_classes, seq_len
            loss = float(loss(scores, rand_pred[:, 1:], ignore_index=-99))
            losses.append(loss)

        losses = torch.stack(losses, dim=1)
        m, argminloss = torch.min(losses, dim=1)
        prediction = [prediction[idx_best][i] for i, idx_best in enumerate(argminloss)]
        return prediction


    # def old_batch(self, data_batch):
    #     scores, _, out_lengths = model(data_batch)
    #     out_lengths = torch.Tensor(out_lengths)
    #     scores = torch.transpose(scores, 0, 1)
    #     probs = F.softmax(scores, dim=2).data.cpu()
    #     output, scores, timesteps, out_seq_len = self.decoder.decode(probs=probs, seq_lens=out_lengths)
    #     out_seq = []
    #     for i in range(output.size(0)):
    #         chrs = [character_list.LETTERS[o.item() - 1] for o in output[i, 0, :out_seq_len[i, 0]]]
    #         out_seq.append("".join(chrs))
    #     return out_seq


def write_results(results):
    with open('predictions.csv', 'w') as f:
        f.write('Id,Predicted\n')
        for i, r in enumerate(results):
            f.write(','.join([str(i), r]))
            f.write('\n')


if __name__ == '__main__':
    import os.path
    import logger

    tLog, vLog = logger.Logger("./logs/train_pytorch"), logger.Logger("./logs/val_pytorch")

    NUM_EPOCHS = 1
    BATCH_SIZE = 64

    model = LAS(num_chars=32, key_size=128, value_size=256, encoder_depth=3, decoder_depth=4, encoder_hidden=512,
                 decoder_hidden=512, cnn_compression=2, enc_bidirectional=True, teacher=0.0)

    def load_my_state_dict(net, state_dict):
        own_state = net.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
        return net


    ckpt_path = 'models/best.pt'
    if os.path.isfile(ckpt_path):
        pretrained_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        model = load_my_state_dict(model, pretrained_dict)
        print('Checkpoint weights loaded.')

    utdst = UtteranceDataset(data_path='./data/train.npy', label_path='./data/train_transcripts.npy')
    val_utdst = UtteranceDataset(data_path='./data/dev.npy', label_path='./data/dev_transcripts.npy')
    test_utdst = UtteranceDataset('./data/test.npy', test=True)
    loader = DataLoader(dataset=utdst, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, num_workers=6)
    val_loader = DataLoader(dataset=val_utdst, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=6)
    test_loader = DataLoader(dataset=test_utdst, batch_size=1, shuffle=False, collate_fn=collate, num_workers=1)

    trainer = LanguageModelTrainer(model=model, loader=loader, val_loader=val_loader,
                                   test_loader=test_loader, max_epochs=NUM_EPOCHS)

    trainer.train()
    write_results(trainer.test())
