import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

# Relational-GCN on heterograph
class HeteroRGCNLayer(nn.Module):
  """
  A layer for Hetero Relational GCN on heterograph

  Parameters
  ----------
  in_size: int
    Input size of the layer.

  out_size: int
    Output size of the layer.

  etypes: list
    List of edge/relation types in the heterograph.
  """
  def __init__(self, in_size, out_size, etypes):
    super(HeteroRGCNLayer, self).__init__()
    # W_r for each relation
    self.weight = nn.ModuleDict({
      name: nn.Linear(in_size, out_size) for name in etypes
    })

  def forward(self, G, feat_dict):
    """Conduct graph convolution on heterograph, according to each relationship.
    
    Parameters
    ----------
    G: DGL Heterograph
      Graph data structure, including different types of nodes and relationships.

    feat_dict: dict
      Dictionary of features for each type of nodes.
      Each features has shape of (n_nodes, in_size).

    Returns
    -------
    dict: dict
      Dictionary of features for each type of nodes after convolution.
      Each features has shape of (n_nodes, out_size).
    """
    funcs = {}
    
    for srctype, etype, dsttype in G.canonical_etypes:
      # Compute W_r * h
      Wh = self.weight[etype](feat_dict[srctype])
      # Save it in graph for message passing
      G.nodes[srctype].data['Wh_%s' % etype] = Wh
      
      # Specify per-relation message passing function: (message_func, reduce_func).
      # Note that the results are saved to the same destination feature 'h',
      # which hints the type wise reducer for aggregation.
      funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
    
    # Trigger message passing of multiple types.
    # The first argument is the message passing functions for each relation.
    # The second one is the type wise reducer, could be "sum", "max", "min", "mean", "stack".
    G.multi_update_all(funcs, 'sum')
    
    # Return the updated node feature dictionary
    return {ntype: G.nodes[ntype].data['h'] for ntype in G.ntypes}

class HeteroRGCN(nn.Module):
  """
  Hetero Relational Graph convolution network model.

  Parameters
  ----------
  G: DGL Heterograph
    Graph data structure, including different types of nodes and relationships.
  
  in_size: int
    Size of the input layer.

  hidden_size: int
    Size of the hidden layer.

  out_size: int
    Size of the output layer.
  """
  def __init__(self, G, in_size, hidden_size, out_size):
    super(HeteroRGCN, self).__init__()
    # Use trainable node embeddings as featureless inputs.
    embed_dict = {ntype: nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), in_size)) for ntype in G.ntypes}
    for key, embed in embed_dict.items():
      nn.init.xavier_uniform_(embed)
    self.embed = nn.ParameterDict(embed_dict)
    # Create layers
    self.layer1 = HeteroRGCNLayer(in_size, hidden_size, G.etypes)
    self.layer2 = HeteroRGCNLayer(hidden_size, out_size, G.etypes)

  def forward(self, G):
    """
    Forward input through the model to produce presentation of patient nodes.

    Parameters
    ----------
    G: DGL Heterograph
      Graph data structure, including different types of nodes and relationships.

    Returns
    -------
    output: tensor shape of (n_patient_nodes, out_size)
      Presentation/Probabilities of patient nodes.
    """
    h_dict = self.layer1(G, self.embed)
    h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
    h_dict = self.layer2(G, h_dict)
      
    # Get patient logits
    return h_dict['patient']