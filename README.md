# COVID-19 patients classification using Graph neural network on a Heterogeneous graph
## Introduction
This repository contains code for node classification on a heterogeneous graph, concretely, patient node classification on a COVID-19 graph.

## Requirements
```bash
scipy
numpy
pandas
pytorch==1.6.0
dgl==0.4.3post2
```

## Installation
- Clone the repository and install the package
```
git clone https://github.com/KienMN/COVID-19-in-Korea-graph.git
cd STGNN-for-Covid-in-Korea
pip install -e .
```
- Install package using `pip`
```
pip install git+https://github.com/KienMN/COVID-19-in-Korea-graph.git
```

## Main components
### Preprocessing
Process dataset from CSV file to DGL graph data structure. Check `graph_neural_network/preprocessing.py` for more details.

### Models
This module contains Relational Graph convolution network model for Heterogeneous graph which conducts graph convolution on each relationship. Check `graph_neural_network/models.py` for more details.

## Citation
This source code is for the paper:

Kien Mai Ngoc, Minho Lee. "COVID-19 patients classification using Graph neural network on a Heterogeneous graph".
In: Proc. of the International Conference on Convergence Content 2020, The Korea Contents Society, 2020, 13-14.
URL: https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE10506109

Bibtex:
```
@inproceedings{mai2020graph,
  author={Kien, Mai Ngoc and Minho, Lee},
  title={COVID-19 patients classification using Graph neural network on a Heterogeneous graph},
  booktitle={Proc. of the International Conference on Convergence Content 2020},
  year={2020},
  pages={13--14},
  publisher={The Korea Contents Society},
  url={https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE10506109}
}
```

## References
1. DGL Documentation: https://docs.dgl.ai/en/0.4.x/tutorials/basics/5_hetero.html
2. COVID-19 in Korea dataset: https://www.kaggle.com/kimjihoo/coronavirusdataset