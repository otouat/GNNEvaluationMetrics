import networkx as nx
import numpy as np
import torch
from prdc import compute_prdc
from scipy import linalg


## Coming from https://github.com/mseitzer/pytorch-fid
def compute_FID(mu1, mu2, cov1, cov2, eps=1e-6):
    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert cov1.shape == cov2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(cov1.dot(cov2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(cov1.shape[0]) * eps
        covmean = linalg.sqrtm((cov1 + offset).dot(cov2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(cov1) +
            np.trace(cov2) - 2 * tr_covmean)


def compute_gnn_metrics(ref_graph, pred_graph, model):
    with torch.no_grad():
        embed_graphs_ref = model.compute_embed_concat(ref_graph)
        embed_graphs_ref = embed_graphs_ref.cpu().detach().numpy()
        mu_ref = np.mean(embed_graphs_ref, axis=0)
        cov_ref = np.cov(embed_graphs_ref, rowvar=False)

        embed_graphs_pred = model.compute_embed_concat(pred_graph)
        embed_graphs_pred = embed_graphs_pred.cpu().detach().numpy()
        mu_pred = np.mean(embed_graphs_pred, axis=0)
        cov_pred = np.cov(embed_graphs_pred, rowvar=False)

        # Fréchet Distance
        fid = compute_FID(mu_ref, mu_pred, cov_ref, cov_pred)
        # Improved Precision and Recall metrics
        prdc = compute_prdc(real_features=embed_graphs_ref,
                            fake_features=embed_graphs_pred,
                            nearest_k=5)
        # "Kernel Inception Distance" : MMD
        # kd,std = kernel_classifier_distance_and_std_from_activations(real_activations=embed_graphs_ref,generated_activations=embed_graphs_pred,dtype="float32")
    return fid, prdc


def test_acc(model, device, train_graph):
    model.eval()

    output = pass_data_iteratively(model, train_graph)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in train_graph]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc = correct / float(len(train_graph))

    print("accuracy : %f" % (acc))

    return acc


def pass_data_iteratively(model, graphs, minibatch_size=64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i + minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)

