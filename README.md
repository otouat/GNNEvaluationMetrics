# GNNEvaluationMetrics
The source code to reproduce results from a study on comparison of GRAN and GraphRNN using new evaluation techniques by training a Graph Neural Network on a graph classification task. 


# Reproducibility
Reproduce the environment with *conda*:
```bash
conda env create -f environment.yml
conda activate sur-gnn-metrics
jupyter notebook
```

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
## Setup & Requirements

This repo uses Python 3.7 with Pytorch 1.8.1
Please use Anaconda to set up the environment and install dependencies using the following command :
```conda create --name <env> --file requirements.txt```

## Run Demos

* To run the main workflow of experiments please follow and run the juptyer notebook file ```GNN_metrics_exp.ipynb```

## Graph Datasets

* Some of the graph datasets were obtained by training GRAN and GraphRNN models ( see the following repositories https://github.com/JiaxuanYou/graph-generation, https://github.com/lrjconan/GRAN )
