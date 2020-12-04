from graph_neural_network.preprocessing import get_heterogeneous_graph
from graph_neural_network.models import HeteroRGCN
import dgl
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

G, patient_idx, labels = get_heterogeneous_graph(include_rest=True)

labels = torch.tensor(labels)
n_labels = len(torch.unique(labels)) - 1

print(n_labels)
# print(patient_idx)
# print(G)
# print(torch.unique(labels, return_counts=True))

np.random.seed(seed=42)
shuffle = np.random.permutation(patient_idx)
train_idx = torch.tensor(shuffle[0: 2250]).long()
val_idx = torch.tensor(shuffle[2250: 3050]).long()
test_idx = torch.tensor(shuffle[3050:]).long()

print("Number of train:", len(train_idx))
print("Number of valid:", len(val_idx))
print("Number of test:", len(test_idx))

device = torch.device('cuda', 0) if torch.cuda.is_available else torch.device('cpu')
print(device)

model = HeteroRGCN(G, 50, 50, n_labels)
opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

best_val_acc = torch.tensor(0).float()
best_test_acc = torch.tensor(0).float()

history = {
    'loss': [],
    'val_loss': [],
    'acc': [],
    'val_acc': [],
    'test_acc': []
}

model = model.to(device)
G = G.to(device)
labels = labels.to(device)

for epoch in range(100):
  logits = model(G)
  # The loss is computed only for labeled nodes.
  loss = F.cross_entropy(logits[train_idx], labels[train_idx])
  val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])

  pred = logits.argmax(1)
  train_acc = (pred[train_idx] == labels[train_idx]).float().mean()
  val_acc = (pred[val_idx] == labels[val_idx]).float().mean()
  test_acc = (pred[test_idx] == labels[test_idx]).float().mean()

  if best_val_acc.item() < val_acc.item():
    best_val_acc = val_acc
    best_test_acc = test_acc
      
  opt.zero_grad()
  loss.backward()
  opt.step()

  history['loss'].append(loss.item())
  history['val_loss'].append(val_loss.item())
  history['acc'].append(train_acc.item())
  history['val_acc'].append(val_acc.item())
  history['test_acc'].append(test_acc.item())
  
  if epoch % 5 == 0:
    print('Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)' % (loss.item(),
                                                                                              train_acc.item(),
                                                                                              val_acc.item(),
                                                                                              best_val_acc.item(),
                                                                                              test_acc.item(),
                                                                                              best_test_acc.item()))