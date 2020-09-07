import torch
import torch.nn as nn
from DKT.dkt_utils import *


class MODEL(nn.Module):

    def __init__(self, feature_dim, n_question, x_embed_dim, hidden_dim, hidden_layers, dropout_rate=0.6, gpu=0):
        super(MODEL, self).__init__()
        self.feature_dim = feature_dim
        self.n_question = n_question
        self.x_embed_dim = x_embed_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.gpu = gpu
        self.hidden_layers = hidden_layers

        self.x_embed = nn.Embedding(4 * self.n_question + 1, self.x_embed_dim, padding_idx=0)

        self.x_linear = nn.Linear(self.feature_dim, self.x_embed_dim, bias=True)

        self.rnn = nn.LSTM(input_size=self.x_embed_dim, hidden_size=self.hidden_dim,
                           num_layers=hidden_layers, batch_first=True)

        self.predict_linear = nn.Linear(self.hidden_dim, self.n_question + 1, bias=True)
        self.dropout = nn.Dropout(dropout_rate)

    def init_params(self):
        nn.init.kaiming_normal_(self.predict_linear.weight)
        nn.init.constant_(self.predict_linear.bias, 0)

    def init_embeddings(self):
        nn.init.kaiming_normal_(self.x_embed.weight)

    def forward(self, x_data, q_t_data, target, loss_fn, q_c_data, c_target):
        # Target size [B*L,1]
        # q_t size [B*L,1]

        batch_size = x_data.shape[0]
        seqlen = x_data.shape[1]

        init_h = variable(torch.randn(self.hidden_layers, batch_size, self.hidden_dim), self.gpu)
        init_c = variable(torch.randn(self.hidden_layers, batch_size, self.hidden_dim), self.gpu)

        ## (q,a) embedding
        # x_embed_data = self.x_embed(x_data)

        # modify
        x_data = x_data.view(batch_size * seqlen, -1)
        x_data = self.x_linear(x_data)
        x_data = x_data.view(batch_size, seqlen, -1)

        ## lstm process
        lstm_out, final_status = self.rnn(x_data, (init_h, init_c))

        ## lstm out size [B,L,E]
        ## Target size [B,L]
        ## prediction size [B*L,Q]
        lstm_out = lstm_out.contiguous()
        # prediction = self.predict_linear(self.dropout(lstm_out.views(batch_size * seqlen, -1)))
        prediction = self.predict_linear(lstm_out.view(batch_size * seqlen, -1))
        # pred_out = torch.relu(prediction)
        # pred_out = torch.sigmoid(prediction)
        pred_out = torch.tanh(prediction)

        c_mask = q_c_data.gt(0)
        c_filtered_pred = torch.masked_select(pred_out, c_mask)
        c_filtered_target = torch.masked_select(c_target, c_mask)
        r = loss_fn(c_filtered_pred, c_filtered_target)

        mask = q_t_data.gt(0)
        filtered_pred = torch.masked_select(pred_out, mask)
        filtered_target = torch.masked_select(target, mask)

        out = pred_out.view(batch_size, seqlen, -1)
        w1 = variable(torch.zeros((batch_size, self.n_question + 1)), self.gpu)
        w2 = variable(torch.zeros((batch_size, self.n_question + 1)), self.gpu)
        for i in range(out.shape[1] - 1):
            d = out[:, i + 1, :] - out[:, i, :]
            w1 += abs(d)
            w2 += d * d
        w1 = torch.sum(w1, 1) / (self.n_question + 1)
        w1 = torch.sum(w1, 0) / batch_size
        w2 = torch.sum(w2, 1) / (self.n_question + 1)
        w2 = torch.sum(w2, 0) / batch_size

        # loss = loss_fn(filtered_pred, filtered_target) + 0.1 * r + 0.003 * w1 + 3.0 * w2
        loss = loss_fn(filtered_pred, filtered_target)

        return loss, filtered_pred, filtered_target, pred_out
