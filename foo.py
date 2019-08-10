import torch


def test():
    with open("model/textCNNModel.pt", 'rb') as f:
        model = torch.load(f, map_location='cpu')
    model.eval()
    inputs = [0, 4, 25164, 1, 1]
    inputs = torch.LongTensor(inputs).view(1, -1)
    # targets = torch.LongTensor([random.randint(0, 1) for _ in range(1)])
    outputs = model(inputs)
    print(outputs, torch.argmax(outputs, 1).item())
    # loss = torch.nn.functional.cross_entropy(outputs, targets)
    # corrects = (torch.max(outputs, 1)[1].view(targets.size()).data == targets.data).sum()


if __name__ == '__main__':
    test()
    # with open("data/vocab.pkl", "rb") as f:
    #     vocab = pickle.load(f)
    # itos=vocab.itos
    # stoi=vocab.stoi
    # print(type(itos), type(stoi))
    # text = "车身尺寸小，找车位太方便了，基本随便一个小空就能停。操控灵活，转弯半径超小。外观新颖可爱，回头率不比百万级别差。"
    # text = re.compile(r'[^A-Za-z0-9\u4e00-\u9fa5]').sub(' ', text)  # 将非中文字符、非a-z, 非A-Z，非0-9 全部替换为' '
    # words = [word.strip() for word in jieba.cut(text) if word.strip()]
    # print(words)
    # d=dict([(1,1),(2,2)])
    # print([i for i in d.items()])
