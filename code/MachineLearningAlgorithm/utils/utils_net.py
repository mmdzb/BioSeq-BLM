import numpy as np
import torch
import torch.nn.functional as func
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pytorch_pretrained_bert import BertModel, BertTokenizer


from .utils_former import Transformer, Reformer

# 需要GPU时改为cuda:0，由于个人笔记本显存不足，只能使用CPU计算 by wzb at 3.24
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
FORMER = ['Transformer', 'Weighted-Transformer', 'Reformer']
torch.set_printoptions(threshold=np.inf)
# print(DEVICE)


# 新增Bert 暂无法使用 by wzb at 3.25
# class BertRes(nn.Module):
#     def __init__(self, hidden_dim, n_classes):
#         super(BertRes, self).__init__()
#         self.bert = BertModel.from_pretrained("./bert_pretrain")
#         for param in self.bert.parameters():
#             param.requires_grad = True
#         self.fc = nn.Linear(hidden_dim, n_classes)
#
#     def forward(self, inputs, seq_mask):
#         _, pooled = self.bert(inputs, attention_mask=seq_mask, output_all_encoded_layers=False)
#         out = self.fc(pooled)
#         return out

# 新增FastText 无法使用 by wzb at 3.26
# class FastTextRes(nn.Module):
#     def __init__(self, in_dim, hidden_dim, n_classes, prob=0.6):
#         super(FastTextRes, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.fc1 = nn.Linear(in_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, n_classes)
#         # self.classifier = nn.Sequential(
#         #     nn.Linear(2 * hidden_dim, hidden_dim),
#         #     nn.ReLU(),
#         #     nn.Linear(hidden_dim, n_classes)
#         # )
#         # self.softmax = nn.Softmax()
#
#     def forward(self, x):
#         h = self.fc1(x.mean(1))
#         z = self.fc2(h)
#         return z

# 定义 Recurrent Network 模型
# class LSTMSeq(nn.Module):
#     def __init__(self, in_dim, hidden_dim, n_layer, n_classes, prob=0.6):
#         super(LSTMSeq, self).__init__()
#         self.n_layer = n_layer
#         self.hidden_dim = hidden_dim
#         self.rnn = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True, dropout=prob, bidirectional=True)
#         self.classifier = nn.Sequential(
#             nn.ReLU(),
#             nn.Linear(2 * hidden_dim, n_classes)
#         )
#
#     def forward(self, x):  # torch.Size([50, 20, 4])  (batch, seq_len, input_size) --> batch_first=True
#         out, _ = self.rnn(x)  # [b_size, len, 2*hidden_dim]
#         out = out[:, -1, :]  # [b_size, 2*hidden_dim]
#         out = self.classifier(out)  # [b_size, num_classes]
#         return out

# my lstm_seq by wzb at 3.29
class LSTMSeq(nn.Module):
    def __init__(self, in_size, hidden_size, num_of_layers, out_size, dropout=0.6):
        super(LSTMSeq, self).__init__()
        # self.lstm = nn.LSTM(in_size, hidden_size, num_of_layers, batch_first=True, dropout=dropout, bidirectional=True)
        # self.classifier = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(2*hidden_size, out_size)
        # )
        self.l1 = nn.Linear(275*4, hidden_size)
        self.active = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        # out, _ = self.lstm(x)
        # out = out[:, -1, :]
        # out = self.classifier(out)
        x = x.view(([x.size()[0], -1]))
        # print(x[0, :, :])
        # print(input_data[0])
        out = self.l1(x)
        out = self.active(out)
        out = self.l2(out)
        return out


# 残基层面的逻辑不一样，应该单拉出来写
class LSTMRes(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_classes, prob=0.5):
        super(LSTMRes, self).__init__()
        self.n_layers = n_layer
        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True, dropout=prob, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, x):
        # [batch_size, seq_len, in_dim]
        x = x
        out, hc = self.rnn(x)  # [batch_size, seq_len, 2*hidden_dim]
        out = self.classifier(out)  # [batch_size, seq_len, num_classes]
        return out


class GRUSeq(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_classes, prob=0.6):
        super(GRUSeq, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(in_dim, hidden_dim, n_layer, batch_first=True, dropout=prob, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, n_classes)
        )

    def forward(self, x):  # torch.Size([50, 20, 4])  (batch, seq_len, input_size) --> batch_first=True
        out, _ = self.rnn(x)  # torch.Size([50, 20, 128])
        out = out[:, -1, :]  # [b_size, 2*hidden_dim]
        out = self.classifier(out)  # [b_size, num_classes]
        return out


# 残基层面的逻辑不一样，单拉出来写
class GRURes(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_classes, prob=0.5):
        super(GRURes, self).__init__()
        self.n_layers = n_layer
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(in_dim, hidden_dim, n_layer, batch_first=True, dropout=prob, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, x):
        # [batch_size, seq_len, in_dim]
        out, hidden = self.rnn(x)  # [batch_size, seq_len, 2*hidden_dim]
        out = self.classifier(out)  # [batch_size, seq_len, num_classes]
        return out


# 这里CNN实现用的二维卷积，但生物信息类似文本，文本中经典的算法是textCNN,使用的一维卷积，这里我也用类似textCNN的模型尝试
class CNNSeq(nn.Module):
    def __init__(self, inchannels, outchannels, kernel_size, n_classes, prob=0.5):
        super(CNNSeq, self).__init__()
        padding = (kernel_size - 1) // 2

        # bn操作 by wzb at 3.29
        self.bn_input = nn.BatchNorm1d(275, momentum=0.5)

        self.cnn = nn.Conv1d(in_channels=inchannels,
                             out_channels=outchannels,
                             kernel_size=kernel_size,
                             padding=padding,
                             stride=1)
        self.dropout = nn.Dropout(prob)
        self.bn = nn.BatchNorm1d(outchannels, momentum=0.5)
        self.classifier = nn.Sequential(
            nn.Linear(outchannels, 2 * outchannels),
            nn.ReLU(),
            nn.Linear(2 * outchannels, n_classes)
        )

    def forward(self, x):
        x = self.bn_input(x)
        input_data = x.permute(0, 2, 1)
        output = self.cnn(input_data)
        output = self.bn(output)
        output = func.max_pool1d(output, kernel_size=output.shape[2])
        output = output.transpose(1, 2).contiguous()
        output = output.view(output.shape[0], -1)
        output = self.dropout(output)
        output = self.classifier(output)
        return output


# 定义CNN， CNN的实现有点问题,
class CNNRes(nn.Module):
    def __init__(self, inchannels, outchannels, kernel_size, n_classes, prob):
        super(CNNRes, self).__init__()
        padding = (kernel_size - 1) // 2
        self.cnn = nn.Conv1d(in_channels=inchannels,
                             out_channels=outchannels,
                             kernel_size=kernel_size,
                             padding=padding,
                             stride=1)
        self.dropout = nn.Dropout(prob)
        self.classifier = nn.Sequential(
            nn.Linear(outchannels, 2 * outchannels),
            nn.ReLU(),
            nn.Linear(2 * outchannels, n_classes)
        )

    def forward(self, x):
        # [batch_size , seq_len, inchannels]
        input_data = x.permute(0, 2, 1)
        output = self.cnn(input_data)
        output = output.transpose(1, 2).contiguous()
        output = self.dropout(output)
        output = self.classifier(output)
        return output


# 定义Transformer和WeightedTransformer序列层面的类
class TransformerSeq(nn.Module):
    def __init__(self, fixed_len, feature_dim, n_layers, d_k, d_v, d_model, d_ff, n_heads, n_classes, dropout=0.1,
                 weighted=False):
        super(TransformerSeq, self).__init__()
        self.transformer = Transformer(feature_dim, n_layers, d_k, d_v, d_model, d_ff, n_heads, dropout, weighted)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * fixed_len, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_classes)
        )

    def forward(self, inputs, seq_mask, mask=False, return_attn=False):
        # inputs = inputs.view(inputs.size()[0], -1)
        output, _ = self.transformer(inputs, seq_mask, mask, return_attn)
        # 注意这里的self-attention不是取最后一个，因为self-attention和LSTM其实不太一样，这里直接将self-attention转变大小考虑了，因此初始化时多了一个fixed_len参数
        output = output.view(output.shape[0], -1)
        # print('经过transformer模型的大小', output.shape)
        output = self.classifier(output)
        # print('经过分类模型的大小', output.shape)
        return output


# 定义Transformer和WeightedTransformer残基层面的类
class TransformerRes(nn.Module):
    def __init__(self, feature_dim, n_layers, d_k, d_v, d_model, d_ff, n_heads, n_classes, dropout=0.1, weighted=False):
        super(TransformerRes, self).__init__()
        self.transformer = Transformer(feature_dim, n_layers, d_k, d_v, d_model, d_ff, n_heads, dropout, weighted)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.ReLU(),
            nn.Linear(2 * d_model, n_classes)
        )

    def forward(self, inputs, seq_mask, mask=True, return_attn=False):
        output, _ = self.transformer(inputs, seq_mask, mask, return_attn)
        # print('经过transformer模型的大小', output.shape)
        output = self.classifier(output)
        # print('经过分类模型的大小', output.shape)
        return output


# 定义Reformer序列层面的类
class ReformerSeq(nn.Module):
    def __init__(self, n_classes, fixed_len, d_model, d_ff, n_heads, n_chunk, rounds, bucket_length, n_layer,
                 dropout_prob=0.1):
        super(ReformerSeq, self).__init__()
        self.reformer = Reformer(d_model, d_ff, n_heads, n_chunk, rounds, bucket_length, n_layer, dropout_prob)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * fixed_len, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_classes)
        )

    def forward(self, inputs, seq_mask, mask=False, return_attn=False):
        # inputs = inputs.view(inputs.size()[0], -1)
        output = self.reformer(inputs, inputs)
        # 注意这里的self-attention不是取最后一个，因为self-attention和LSTM其实不太一样，
        # 这里直接将self-attention转变大小考虑了，因此初始化时多了一个fixed_len参数
        output = output.view(output.shape[0], -1)
        # print('经过transformer模型的大小', output.shape)
        output = self.classifier(output)
        # print('经过分类模型的大小', output.shape)
        return output


# 定义Transformer和WeightedTransformer残基层面的类
class ReformerRes(nn.Module):
    def __init__(self, n_classes, d_model, d_ff, n_heads, n_chunk, rounds, bucket_length, n_layer, dropout_prob=0.1):
        super(ReformerRes, self).__init__()
        self.reformer = Reformer(d_model, d_ff, n_heads, n_chunk, rounds, bucket_length, n_layer, dropout_prob)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.ReLU(),
            nn.Linear(2 * d_model, n_classes)
        )

    def forward(self, inputs):
        output = self.reformer(inputs, inputs)
        # print('经过transformer模型的大小', output.shape)
        output = self.classifier(output)
        # print('经过分类模型的大小', output.shape)
        return output


class MyDataset(Dataset):
    def __init__(self, feature, target, length, max_len):
        self.feature = feature
        self.target = target
        self.length = length
        self.max_len = max_len

    def __getitem__(self, index):
        return self.feature[index], self.target[index], np.array([self.length[index]], dtype=np.int), self.max_len

    def __len__(self):
        return len(self.feature)


def batch_seq(data):
    inputs = []
    inputs_length = []
    input_mask = []
    targets = []
    for feature, target, length, max_len in data:
        inputs.append(torch.FloatTensor(feature).unsqueeze(0))  # [fixed_Len, fea_dim]
        # print(length)
        inputs_length.append(torch.LongTensor(np.array(length, dtype=np.int)))  # [1]
        # print(inputs_length)
        mask = sequence_mask(torch.LongTensor(np.array(length, dtype=np.int)), max_len)
        input_mask.append(torch.FloatTensor(mask))  # [1, max_len]
        targets.append(torch.LongTensor(np.array([target], dtype=np.int)))  # [1]

    inputs = torch.cat(inputs)
    # print(inputs.shape)
    inputs_length = torch.cat(inputs_length)
    # print(inputs_length)
    input_mask = torch.cat(input_mask)
    # print(input_mask.shape)
    targets = torch.cat(targets)
    # print(targets.shape)

    return inputs, inputs_length, input_mask, targets


def batch_res(data):
    inputs = []
    inputs_length = []
    input_mask = []
    targets = []
    for feature, target, length, max_len in data:
        inputs.append(torch.FloatTensor(feature).unsqueeze(0))  # [fixed_Len, fea_dim]
        # print(length)
        inputs_length.append(torch.LongTensor(np.array(length, dtype=np.int)))  # [1]
        # print(inputs_length)
        mask = sequence_mask(torch.LongTensor(np.array(length, dtype=np.int)), max_len)
        input_mask.append(torch.FloatTensor(mask))  # [1, max_len]
        targets.append(torch.LongTensor(np.array(target, dtype=np.int)).unsqueeze(0))  # [fixed_len, fea_dim]
        # print(targets)
    inputs = torch.cat(inputs)
    # print(inputs.shape)
    inputs_length = torch.cat(inputs_length)
    # print(inputs_length)
    input_mask = torch.cat(input_mask)
    # print(input_mask.shape)
    targets = torch.cat(targets)
    # print(targets.shape)

    return inputs, inputs_length, input_mask, targets


def sequence_mask(lengths, max_len):
    """
    lengths: [len1, len2....] 一个长度为batch的包含序列长度的列表
    max_len: 也就是红良的fixed_len
    返回值: [batch_size, seq_len]
    """
    batch_size = lengths.numel()
    max_len = max_len
    mask = torch.arange(0, max_len).type_as(lengths).unsqueeze(0).expand(batch_size, max_len).lt(
        lengths.unsqueeze(1)).float()
    return mask


def criterion_func(inputs, targets, seq_length, mask):
    inputs = func.softmax(inputs, dim=-1)  # [batch_size, seq_len, 2]
    one_hot = func.one_hot(targets, num_classes=inputs.size()[-1]).float()
    inputs = torch.clamp(inputs, min=1e-8, max=1)
    prob = torch.sum(torch.log(inputs) * one_hot, dim=-1)  # [batch_size, seq_len]

    log_prob = -prob * mask
    log_prob = log_prob.view(-1)  # [batch_size*seq_len]
    loss = torch.sum(log_prob) / torch.sum(seq_length)

    return loss


class TorchNetSeq(object):
    def __init__(self, net, max_len, criterion, params_dict):
        super(TorchNetSeq, self).__init__()
        self.net = net
        self.dropout = params_dict['dropout']
        self.batch_size = params_dict['batch_size']
        self.max_len = max_len
        self.criterion = criterion
        self.params_dict = params_dict

    def prepare(self, data, labels, input_length, shuffle=True):
        dataset = MyDataset(data, labels, input_length, self.max_len)
        data_iter = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, collate_fn=batch_seq)
        return data_iter

    def net_type(self, in_dim, n_classes):
        if self.net == 'LSTM':
            hidden_dim = self.params_dict['hidden_dim']
            n_layer = self.params_dict['n_layer']

            model = LSTMSeq(in_dim, hidden_dim, n_layer, n_classes, self.dropout).to(DEVICE)
        elif self.net == 'GRU':
            hidden_dim = self.params_dict['hidden_dim']
            n_layer = self.params_dict['n_layer']
            model = GRUSeq(in_dim, hidden_dim, n_layer, n_classes, self.dropout).to(DEVICE)
        elif self.net == 'CNN':
            out_channels = self.params_dict['out_channels']
            kernel_size = self.params_dict['kernel_size']
            model = CNNSeq(in_dim, out_channels, kernel_size, n_classes, self.dropout).to(DEVICE)
        elif self.net == 'Transformer':
            n_layer = self.params_dict['n_layer']
            d_model = self.params_dict['d_model']
            d_ff = self.params_dict['d_ff']
            n_heads = self.params_dict['n_heads']
            model = TransformerSeq(self.max_len, in_dim, n_layer, d_model, d_model, d_model,
                                   d_ff, n_heads, n_classes, self.dropout, False).to(DEVICE)
        elif self.net == 'Weighted-Transformer':
            n_layer = self.params_dict['n_layer']
            d_model = self.params_dict['d_model']
            d_ff = self.params_dict['d_ff']
            n_heads = self.params_dict['n_heads']
            model = TransformerSeq(self.max_len, in_dim, n_layer, d_model, d_model, d_model,
                                   d_ff, n_heads, n_classes, self.dropout, True).to(DEVICE)
        else:
            d_model = self.params_dict['d_model']
            d_ff = self.params_dict['d_ff']
            n_heads = self.params_dict['n_heads']
            n_chunk = self.params_dict['n_chunk']
            rounds = self.params_dict['rounds']
            bucket_length = self.params_dict['bucket_length']
            n_layer = self.params_dict['n_layer']
            model = TransformerSeq(n_classes, self.max_len, d_model, d_ff, n_heads, n_chunk, rounds, bucket_length,
                                   n_layer, self.dropout).to(DEVICE)
        return model

    def train(self, model, optimizer, train_x, train_y, train_len_list, epoch):
        # in_dim = train_x.shape[-1]
        model.train()
        train_loader = self.prepare(train_x, train_y, train_len_list)

        for batch_idx, (inputs, inputs_length, input_mask, target) in enumerate(train_loader):

            torch.cuda.empty_cache()
            inputs = inputs.cuda()
            inputs_length = inputs_length.cuda()
            input_mask = input_mask.cuda()
            target = target.cuda()

            if self.net in FORMER:
                output = model(inputs, input_mask)
            else:
                output = model(inputs)
            # 为实现二分类任务，由softmax修改为sigmoid by wzb at 3.29

            # target = torch.unsqueeze(target, 1)
            # target = torch.cat((target, 1-target), 1)
            # loss = self.criterion(output, target.float())

            loss = self.criterion(output, target)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 20 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(inputs), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    loss.item()))

    def test(self, model, test_x, test_y, test_len_list):
        model.eval()
        test_loss = 0
        correct = 0
        all_num = 0
        predict = []
        prob_list = []
        target_list = []
        test_loader = self.prepare(test_x, test_y, test_len_list, shuffle=False)

        for inputs, inputs_length, input_mask, target in test_loader:

            torch.cuda.empty_cache()
            inputs = inputs.cuda()
            inputs_length = inputs_length.cuda()
            input_mask = input_mask.cuda()
            target = target.cuda()

            if self.net in FORMER:
                output = model(inputs, input_mask)
            else:
                output = model(inputs)
            # 为实现二分类任务，由softmax修改为sigmoid by wzb at 3.29
            # output_1 = output[:, -1:, 1]
            # output_1 = torch.softmax(output, dim=-1)
            # output_1 = torch.max(output_1, dim=-1)
            # test_loss += self.criterion(output_1.values, target.float())
            # output = torch.softmax(output, dim=-1)
            # # output = output[:, -1]
            # predict_label = torch.max(output, dim=-1)[1]

            # target1 = torch.unsqueeze(target, 1)
            # target1 = torch.cat((target1, 1 - target1), 1)
            # test_loss += self.criterion(output, target1.float())
            # output = torch.softmax(output, dim=-1)
            # predict_label = torch.max(output, dim=-1)[1]

            test_loss += self.criterion(output, target)
            output = torch.softmax(output, dim=-1)
            predict_label = torch.max(output, dim=-1)[1]

            num = 0
            for i in range(len(input_mask)):
                prob_list.append(float(output[i][1]))
                predict.append(int(predict_label[i]))
                target_list.append(int(target[i]))
                if predict_label[i] == target[i]:
                    num += 1
            correct += num
            all_num += len(input_mask)

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, all_num,
            100. * correct / all_num))
        return predict, target_list, prob_list, test_loss


class TorchNetRes(object):
    def __init__(self, net, max_len, criterion, params_dict):
        super(TorchNetRes, self).__init__()
        self.net = net
        self.dropout = params_dict['dropout']
        self.batch_size = params_dict['batch_size']
        self.max_len = max_len
        self.criterion = criterion
        self.params_dict = params_dict

    def prepare(self, data, labels, input_length, shuffle=True):
        dataset = MyDataset(data, labels, input_length, self.max_len)
        data_iter = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, collate_fn=batch_res)
        return data_iter

    def net_type(self, in_dim, n_classes):
        if self.net == 'LSTM':
            hidden_dim = self.params_dict['hidden_dim']
            n_layer = self.params_dict['n_layer']
            model = LSTMRes(in_dim, hidden_dim, n_layer, n_classes, self.dropout).to(DEVICE)
        elif self.net == 'GRU':
            hidden_dim = self.params_dict['hidden_dim']
            n_layer = self.params_dict['n_layer']
            model = GRURes(in_dim, hidden_dim, n_layer, n_classes, self.dropout).to(DEVICE)
        elif self.net == 'CNN':
            out_channels = self.params_dict['out_channels']
            kernel_size = self.params_dict['kernel_size']
            model = CNNRes(in_dim, out_channels, kernel_size, n_classes, self.dropout).to(DEVICE)
        elif self.net == 'Transformer':
            n_layer = self.params_dict['n_layer']
            d_model = self.params_dict['d_model']
            d_ff = self.params_dict['d_ff']
            n_heads = self.params_dict['n_heads']
            model = TransformerRes(in_dim, n_layer, d_model, d_model, d_model,
                                   d_ff, n_heads, n_classes, self.dropout, False).to(DEVICE)
        elif self.net == 'Weighted-Transformer':
            n_layer = self.params_dict['n_layer']
            d_model = self.params_dict['d_model']
            d_ff = self.params_dict['d_ff']
            n_heads = self.params_dict['n_heads']
            model = TransformerRes(in_dim, n_layer, d_model, d_model, d_model,
                                   d_ff, n_heads, n_classes, self.dropout, True).to(DEVICE)
        # 添加Bert 暂无法使用 by wzb at 3.25
        # elif self.net == 'Bert':
        #     n_layer = self.params_dict['n_layer']
        #     d_model = self.params_dict['d_model']
        #     d_ff = self.params_dict['d_ff']
        #     n_heads = self.params_dict['n_heads']
        #     model = BertRes(in_dim, n_layer).to(DEVICE)
        #添加FastText by wzb at 3.26
        elif self.net == 'FastText':
            hidden_dim = self.params_dict['hidden_dim']
            model = FastTextRes(in_dim, hidden_dim, n_classes, self.dropout)
        else:
            d_model = self.params_dict['d_model']
            d_ff = self.params_dict['d_ff']
            n_heads = self.params_dict['n_heads']
            n_chunk = self.params_dict['n_chunk']
            rounds = self.params_dict['rounds']
            bucket_length = self.params_dict['bucket_length']
            n_layer = self.params_dict['n_layer']
            model = TransformerRes(n_classes, d_model, d_ff, n_heads, n_chunk, rounds, bucket_length, n_layer,
                                   self.dropout).to(DEVICE)
        return model

    def train(self, model, optimizer, train_x, train_y, train_len_list, epoch):
        # in_dim = train_x.shape[-1]
        model.train()
        train_loader = self.prepare(train_x, train_y, train_len_list)

        for batch_idx, (inputs, inputs_length, input_mask, target) in enumerate(train_loader):

            # 为实现GPU运算，将数据放置于GPU上 by wzb at 3.24
            # torch.cuda.empty_cache()
            # inputs = inputs.cuda()
            # inputs_length = inputs_length.cuda()
            # input_mask = input_mask.cuda()
            # target = target.cuda()

            if self.net in ['LSTM', 'GRU', 'CNN', 'FastText']:
                output = model(inputs)
            else:
                output = model(inputs, input_mask)
            # print(output.size())
            # print(target)
            # print(inputs_length)
            # print(input_mask)
            loss = self.criterion(output, target, inputs_length, input_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 20 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(inputs), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    loss.item()))

    def test(self, model, test_x, test_y, test_len_list):
        model.eval()
        test_loss = 0
        correct = 0
        all_num = 0
        predict = []
        prob_list = []
        target_list = []
        prob_list_format = []
        predict_list_format = []
        test_loader = self.prepare(test_x, test_y, test_len_list, shuffle=False)

        for inputs, inputs_length, input_mask, target in test_loader:

            # 为实现GPU运算，将数据放置于GPU上 by wzb at 3.24
            # torch.cuda.empty_cache()
            # inputs = inputs.cuda()
            # inputs_length = inputs_length.cuda()
            # input_mask = input_mask.cuda()
            # target = target.cuda()

            if self.net in FORMER:
                output = model(inputs, input_mask)
            else:
                output = model(inputs)

            test_loss += self.criterion(output, target, inputs_length, input_mask)
            output = func.softmax(output, dim=-1)
            predict_label = torch.max(output, dim=-1)[1]
            num = 0
            for i in range(len(input_mask)):
                pred_list = []
                prob = []
                for j in range(len(input_mask[i])):
                    if input_mask[i][j] > 0:
                        prob_list.append(float(output[i][j][1]))
                        prob.append(float(output[i][j][1]))
                        pred_list.append(int(predict_label[i][j]))
                        predict.append(int(predict_label[i][j]))
                        target_list.append(int(target[i][j]))
                        if predict_label[i][j] == target[i][j]:
                            num += 1
                prob_list_format.append(prob)
                predict_list_format.append(pred_list)
            correct += num
            all_num += torch.sum(inputs_length)

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, all_num,
            100. * correct / all_num))
        return predict, target_list, prob_list, test_loss, prob_list_format, predict_list_format
