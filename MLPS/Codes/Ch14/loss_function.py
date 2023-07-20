"""
@Description: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-07-20 21:38:28
"""

import torch
import torch.nn as nn
logits = torch.tensor([.8])
probas = torch.sigmoid(logits)
target = torch.tensor([1.0])
bce_loss_fn = nn.BCELoss()
bce_logits_loss_fn = nn.BCEWithLogitsLoss()
print(f'BCE (w Probas): {bce_loss_fn(probas, target):.4f}')
print(f'BCE (w logits): {bce_logits_loss_fn(logits, target):.4f}')


# Categorical cross-entropy
logits = torch.tensor([[1.5, .8, 2.1]])
probas = torch.softmax(logits, dim=1)
target = torch.tensor([2])
cce_loss_fn = nn.NLLLoss()
cce_logits_loss_fn = nn.CrossEntropyLoss()
print(f'CCE (w probas): {cce_logits_loss_fn(logits, target):.4f}')
print(f'CCE (w Logits): {cce_loss_fn(torch.log(probas), target):.4f}')
