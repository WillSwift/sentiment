from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim

from model import Model
from data_utils import DataLoader


def logging(outf, str):
    with open(os.path.join(outf, 'log'), 'a') as f:
        print(str, file=f)
    print(str)


def train(opt, model, dataloader, optimizer):
    train_loss, train_n_correct, train_n = [], 0, 0
    lr_decay_count = best_eval_acc = 0

    for iter in range(opt.max_niter):
        # train with real
        optimizer.zero_grad()
        input, target, input_len =\
                dataloader.get_random_batch(opt.batch_size, set='train')
        input = input.to('cuda')
        target = target.to('cuda')
        input_len = input_len.to('cuda')
        output = model(input, input_len)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        train_n_correct += pred.eq(target.view_as(pred)).sum().item()
        train_n += output.size(0)
        train_loss.append(loss.item() * output.size(0))

        if iter % opt.eval_niter == 0:
            # train logging
            logging(opt.outf, "<<<<<<< Iteration {} >>>>>>>>".format(iter))
            logging(opt.outf, "Training loss: {}, accuracy: {}".format(
                        sum(train_loss) / train_n,
                        train_n_correct / train_n))
            train_loss, train_n_correct, train_n = [], 0, 0

            # eval logging
            eval_loss, eval_n_correct, eval_n = [], 0, 0
            finish_flag = False
            while not finish_flag:
                input, target, input_len, finish_flag =\
                        dataloader.get_iter_batch(opt.batch_size, set='test')
                input = input.to('cuda')
                target = target.to('cuda')
                input_len = input_len.to('cuda')
                output = model(input, input_len)
                loss = F.cross_entropy(output, target)
                pred = output.argmax(dim=1, keepdim=True)
                eval_n_correct += pred.eq(target.view_as(pred)).sum().item()
                eval_n += output.size(0)
                eval_loss.append(loss.item() * output.size(0))

            cur_eval_acc = eval_n_correct / eval_n
            if best_eval_acc < cur_eval_acc:
                lr_decay_count = 0
                best_eval_acc = cur_eval_acc
                torch.save(model.state_dict(),
                        '{}/model_ckp_iter_{}.pth'.format(opt.outf, iter))
            else:
                lr_decay_count += 1
                if lr_decay_count == 3:
                    logging(opt.outf, 'Learning rate decay...')
                    for param_group in optimizer.param_groups:
                        param_group['lr'] /= 2
                    lr_decay_count = 0

            logging(opt.outf, 
                  "Evaluate loss: {}, accuracy: {}, best accuracy: {}".format(
                        sum(eval_loss) / eval_n, cur_eval_acc, best_eval_acc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str,
                        default='/tmp/haitong_ai/sentiment/baidu_sentiment',
                        help='path to dataset')
    parser.add_argument('--outf', type=str, default='./model/',
                        help='path to output folder')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--model-type', type=str, default='lstm',
                        help='type of the top level model: [lstm | gru]')
    parser.add_argument('--dim', type=int, default=768,
                        help='feature layer size')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers at the top level model')
    parser.add_argument('--dropout', type=float, default=0,
                        help='dropout probability.')
    parser.add_argument('--bidir', dest='bidir', action='store_true')
    parser.add_argument('--no-bidir', dest='bidir', action='store_false')
    parser.set_defaults(bidir=True)
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--max_niter', type=int, default=2000,
                        help='number of iters in total')
    parser.add_argument('--eval-niter', type=int, default=100,
                        help='number of iters at each evaluation cycle')

    opt = parser.parse_args()

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass
    logging(opt.outf, opt)

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    logging(opt.outf, "Random Seed: {}".format(opt.manualSeed))
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    # define data loader  TODO
    dataloader = DataLoader(opt.dataroot, None) 
    '''
    for elem in dataloader.test_set:
        counter.append(len(elem['pkl']['NUMBER']) if 'NUMBER' in elem['pkl'] else 0)
    from collections import Counter
    print(sorted(Counter(counter).items()))
    import sys
    sys.exit(1)
    '''

    # training prep
    model = Model(opt.model_type, opt.dim, opt.nlayers, opt.bidir,
                  opt.dropout).to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    train(opt, model, dataloader, optimizer)


if __name__ == '__main__':
    main()
