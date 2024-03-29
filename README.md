# GNNEvaluationMetrics
The source code to reproduce results from a study on comparison of GRAN and GraphRNN using new evaluation techniques by training a Graph Neural Network on a graph classification task. 

You can cite our work:
```bibtex
@inproceedings{10.1007/978-3-031-53969-5_32,
author="Touat, Ousmane and Stier, Julian and Portier, Pierre-Edouard and Granitzer, Michael",
editor="Nicosia, Giuseppe and Ojha, Varun and La Malfa, Emanuele and La Malfa, Gabriele and Pardalos, Panos M.and Umeton, Renato",
title="GRAN Is Superior to GraphRNN: Node Orderings, Kernel- and Graph Embeddings-Based Metrics for Graph Generators",
booktitle="Machine Learning, Optimization, and Data Science",
year="2024",
publisher="Springer Nature Switzerland",
address="Cham",
pages="430--444",
isbn="978-3-031-53969-5"
}
```


# Reproducibility
Reproduce the environment with *conda*, on Linux based machine (tested with Ubuntu 20.04.4 LTS):
```bash
conda env create -f environment.yml
conda activate sur-gnn-metrics
jupyter notebook
```

### Setup & Requirements

This repo uses Python 3.7 with Pytorch 1.8.1 + CUDA 11.3.1 (it can still run on cpu, albeit running slower)

### Run Demos

* To run the main workflow of experiments please follow and run the juptyer notebook file ```GNN_metrics_exp.ipynb```

### Graph Datasets

* Some of the graph datasets were obtained by training GRAN and GraphRNN models ( see the following repositories https://github.com/JiaxuanYou/graph-generation, https://github.com/lrjconan/GRAN )

# References
This is using the official PyTorch implementation of GIN [How Powerful are Graph Neural Networks ?](https://github.com/weihua916/powerful-gnns) as described in the following paper:

```
@misc{xu2019powerful,
      title={How Powerful are Graph Neural Networks?}, 
      author={Keyulu Xu and Weihua Hu and Jure Leskovec and Stefanie Jegelka},
      year={2019},
      eprint={1810.00826},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
