'''
copy from MetaBIN
'''
import torch
import torch.nn.functional as F
from .triplet_loss import euclidean_dist, normalize, cosine_dist, cosine_sim

def domain_SCT_loss(embedding, domain_labels, norm_feat=False, type='cos_sim'):

    # eps=1e-05
    if norm_feat: embedding = normalize(embedding, axis=-1)
    unique_label = torch.unique(domain_labels)
    embedding_all = list()
    for i, x in enumerate(unique_label):
        embedding_all.append(embedding[x == domain_labels])
    num_domain = len(embedding_all)
    loss_all = []
    for i in range(num_domain):
        feat = embedding_all[i]
        center_feat = torch.mean(feat, 0)
        if type == 'euc':
            loss = torch.mean(euclidean_dist(center_feat.view(1, -1), feat))
            loss_all.append(-loss)
        elif type == 'cos':
            loss = torch.mean(cosine_dist(center_feat.view(1, -1), feat))
            loss_all.append(-loss)
        elif type == 'cos_sim':
            loss = torch.mean(cosine_sim(center_feat.view(1, -1), feat))
            loss_all.append(loss)

    loss_all = torch.mean(torch.stack(loss_all))

    return loss_all