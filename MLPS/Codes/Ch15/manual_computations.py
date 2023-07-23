"""
@Description: let’s create the layer and assign the weights and biases for our manual computations:
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-23 21:49:05
"""

import torch
import torch.nn as nn

torch.manual_seed(1)
rnn_layer = nn.RNN(input_size=5, hidden_size=2,
                   num_layers=1, batch_first=True)
w_xh = rnn_layer.weight_ih_l0
w_hh = rnn_layer.weight_hh_l0
b_xh = rnn_layer.bias_ih_l0
b_hh = rnn_layer.bias_hh_l0

print(f'W_xh shape:', {w_xh.shape})
print(f'W_hh shape:', {w_hh.shape})
print(f'b_xh shape:', {b_xh.shape})
print(f'b_hh shape:', {b_hh.shape})

x_seq = torch.tensor([[1.0] * 5, [2.0] * 5, [3.0] * 5]).float()
output, hn = rnn_layer(torch.reshape(x_seq, (1, 3, 5)))
# output, hn = rnn_layer(x_seq.reshape(-1, 3, 5))
out_manual = []
for t in range(3):
    xt = torch.reshape(x_seq[t], (1, 5))
    print(f'Time step {t} =>')
    print(f'Input:{"":10}', xt.numpy())
    ht = torch.matmul(xt, torch.transpose(w_xh, 0, 1)) + b_xh
    print(f'Hidden:{"":9}', ht.detach().numpy())

    if t > 0:
        prev_h = out_manual[t - 1]
    else:
        prev_h = torch.zeros((ht.shape))
    ot = ht + torch.matmul(prev_h, torch.transpose(w_hh, 0, 1)) + b_hh
    ot = torch.tanh(ot)
    out_manual.append(ot)
    print(f'Output (manual):', ot.detach().numpy())
    print(f'RNN output:{"":5}', output[:, t].detach().numpy())
