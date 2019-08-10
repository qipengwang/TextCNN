import argparse
from torchtext.vocab import Vectors
import torch
import torch.nn.functional as F
import re
import sys
from torchtext import data
import jieba
import logging
from model import TextCNN
import pickle


def word_cut(text):
    text = re.compile(r'[^A-Za-z0-9\u4e00-\u9fa5]').sub(' ', text)  # 将非中文字符、非a-z, 非A-Z，非0-9 全部替换为' '
    return [word.strip() for word in jieba.cut(text) if word.strip()]


def load_iter(text_field, label_field, args, **kwargs):
    print("load_iter...")
    text_field.tokenize = word_cut
    train_dataset, test_dataset = data.TabularDataset.splits(
        path='data', format='tsv', skip_header=True,
        train='train.tsv', test='test.tsv',
        fields=[
            ('index', None),
            ('label', label_field),
            ('text', text_field)
        ]
    )
    if args.static and args.pretrained_name and args.pretrained_path:
        vectors = Vectors(name=args.pretrained_name, cache=args.pretrained_path)
        text_field.build_vocab(train_dataset, test_dataset, vectors=vectors)
    else:
        text_field.build_vocab(train_dataset, test_dataset)
    label_field.build_vocab(train_dataset, test_dataset)  # word2index
    train_iter, test_iter = data.Iterator.splits(
        (train_dataset, test_dataset),
        batch_sizes=(args.batch_size, len(test_dataset)),
        sort_key=lambda x: len(x.text),
        **kwargs
    )
    print("finish load_iter")
    return train_iter, test_iter


def load(args):
    print('load_data...')
    # , unk_token=None, pad_token=None
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False, unk_token=None, pad_token=None)
    train_iter, test_iter = load_iter(text_field, label_field, args, device=-1, repeat=False, shuffle=True)
    args.vocabulary_size = len(text_field.vocab)
    # print(label_field.vocab.itos)  # ['<unk>', '0', '1']
    if args.static:
        args.embedding_dim = text_field.vocab.vectors.size()[-1]
        args.vectors = text_field.vocab.vectors
    if args.multichannel:
        args.static = True
        args.non_static = True
    args.class_num = len(label_field.vocab)
    # if '<unk>' in label_field.vocab.itos:
    #     args.class_num -= 1
    # if '<pad>' in label_field.vocab.itos:
    #     args.class_num -= 1
    # Field对象有个specialist对象里面有unk_token默认为<unk>，如果Field的时候没有说明，Vocab就里面会有<unk>
    args.cuda = args.device != -1 and torch.cuda.is_available()
    args.filter_sizes = [int(size) for size in args.filter_sizes.split(',')]
    print('Finish load_data')
    return train_iter, test_iter, text_field.vocab


def set_args():
    parser = argparse.ArgumentParser(description='TextCNN text classifier')
    # learning
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
    parser.add_argument('-batch-size', type=int, default=128, help='batch size for training [default: 128]')
    parser.add_argument('-log-interval', type=int, default=1, help='how many steps to wait before logging training status [default: 1]')
    parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
    parser.add_argument('-early-stopping', type=int, default=1000, help='iteration numbers to stop without performance increasing')
    parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
    # model
    parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
    parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
    parser.add_argument('-embedding-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
    parser.add_argument('-filter-num', type=int, default=100, help='number of each size of filter')
    parser.add_argument('-filter-sizes', type=str, default='3,4,5', help='comma-separated filter sizes to use for convolution')
    parser.add_argument('-static', type=bool, default=False, help='whether to use static pre-trained word vectors')
    parser.add_argument('-non-static', type=bool, default=False, help='whether to fine-tune static pre-trained word vectors')
    parser.add_argument('-multichannel', type=bool, default=False, help='whether to use 2 channel of word vectors')
    parser.add_argument('-pretrained-name', type=str, default='sgns.zhihu.word', help='filename of pre-trained word vectors')
    parser.add_argument('-pretrained-path', type=str, default='pretrained', help='path of pre-trained word vectors')
    # device
    parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
    return parser.parse_args()


def train(train_iter, model, args):
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bestLoss = None
    model.train()
    for epoch in range(1, args.epochs + 1):
        epochLoss = 0
        steps = 0
        for batch in train_iter:
            feature, target = batch.text, batch.label
            # print(feature, target)
            feature.data.t_()  # , target.data.sub_(1)
            # print(target)
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
            optimizer.zero_grad()
            # feature = torch.cat((feature, feature), 0)
            # target = torch.cat((target, target), 0)
            # print(feature.shape, target.shape)  # [128, 55左右], [128]二维feature第0维度和target一样就行
            # break
            logits = model(feature)  # [128, 3]
            # print(target.max())
            # print(logits.shape)
            # break
            # print(logits.sum(1))
            # break
            # print(logits.shape, target)
            loss = F.cross_entropy(logits, target)
            epochLoss += loss.item()
            loss.backward()
            optimizer.step()
            steps += 1
            corrects = (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
            train_acc = 100.0 * corrects / batch.batch_size
            sys.stdout.write('\repoch[{}/{}] batch[{}/{}] - loss: {:.6f}  acc: {}/{}={:.4f}%'.
                             format(epoch, args.epochs, steps, len(train_iter), loss.item(), corrects, batch.batch_size, train_acc))
        # save model
        print("")
        if bestLoss is None or epochLoss < bestLoss:
            print('epoch[{}/{}] save model with loss={}'.format(epoch, args.epochs, epochLoss))
            bestLoss = epochLoss
            torch.save(model, 'model/textCNNModel.pt')
        # update lr
        for param_group in optimizer.param_groups:
            param_group['lr'] -= args.lr / args.epochs


def evaluate(test_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in test_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        logits = model(feature)
        loss = F.cross_entropy(logits, target)
        avg_loss += loss.item()
        corrects += (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
    size = len(test_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, accuracy, corrects, size))
    return accuracy


if __name__ == '__main__':
    jieba.setLogLevel(logging.INFO)
    args = set_args()
    train_iter, test_iter, vocab = load(args)
    with open("data/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    text_cnn = TextCNN(args)
    if args.cuda:
        torch.cuda.set_device(args.device)
        text_cnn = text_cnn.cuda()
    train(train_iter, text_cnn, args)
