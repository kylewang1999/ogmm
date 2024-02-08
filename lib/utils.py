#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/11/2022 10:59 AM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : utils.py
# @Software: PyCharm
import torch
import torch.nn.functional as F


def square_distance(src, dst, normalize=False):
    """
    Calculate Euclid distance between each two src_xyz.
    src_xyz^T * ref_xyz = xn * xm + yn * ym + zn * zm；
    sum(src_xyz^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(ref_xyz^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src_xyz**2,dim=-1)+sum(ref_xyz**2,dim=-1)-2*src_xyz^T*ref_xyz
    Input:
        src_xyz: ref_xyz src_xyz, [B, N, C]
        ref_xyz: target src_xyz, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    if normalize:
        return 2.0 + dist
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    dist = torch.clamp(dist, min=1e-12)
    return dist


def knn(src, tgt, k, normalize=False):
    '''
    Find K-nearest neighbor when ref==ref_xyz and query==src_xyz
    Return index of knn, [B, N, k]
    '''
    dist = square_distance(src, tgt, normalize)
    _, idx = torch.topk(dist, k, dim=-1, largest=False, sorted=True)
    return idx


def get_graph_feature(x, k=20, idx=None, extra_dim=False):
    batch_size, num_dims, num_points = x.size()
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if extra_dim is False:
            idx = knn(x.transpose(-1, -2), x.transpose(-1, -2), k=k)
        else:
            idx = knn(x[:, 6:].transpose(-1, -2), x[:, 6:].transpose(-1, -2), k=k)  # idx = knn(src_xyz[:, :3], k=k)
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx += idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature  # (batch_size, 2 * num_dims, num_points, k)


def log_boltzmann_kernel(cost, u, v, epsilon):
    kernel = (-cost + u.unsqueeze(-1) + v.unsqueeze(-2)) / epsilon
    return kernel


def sinkhorn(cost, p=None, q=None, epsilon=1e-2, thresh=1e-2, max_iter=100):
    ''' Sinkhorn-Knopp algorithm for optimal transport.
    Call stack: GMMReg.forward() -> get_anchor_corrs() -> wkeans() -> sinkhorn()
    
    Input:
        - cost: Cost matrix.
        - p: [B,N=717]. Source distribution.
        - q: [B,J=128]. Target distribution.
    Output:
        - gamma: Optimal transport matrix.
        - loss: Sinkhorn loss.
    '''
    
    if p is None or q is None:
        batch_size, num_x, num_y = cost.shape
        device = cost.device
        if p is None:
            p = torch.empty(batch_size, num_x, dtype=torch.float,
                            requires_grad=False, device=device).fill_(1.0 / num_x).squeeze()
        if q is None:
            q = torch.empty(batch_size, num_y, dtype=torch.float,
                            requires_grad=False, device=device).fill_(1.0 / num_y).squeeze()
    # Initialise approximation vectors in log domain
    u = torch.zeros_like(p).to(p)
    v = torch.zeros_like(q).to(q)
    # Stopping criterion, gmmlib iterations
    for i in range(max_iter):
        u0, v0 = u, v
        # u^{l+1} = a / (K v^l)
        K = log_boltzmann_kernel(cost, u, v, epsilon)
        u_ = torch.log(p + 1e-8) - torch.logsumexp(K, dim=-1)
        u = epsilon * u_ + u
        # v^{l+1} = b / (K^T u^(l+1))
        Kt = log_boltzmann_kernel(cost, u, v, epsilon).transpose(-2, -1)
        v_ = torch.log(q + 1e-8) - torch.logsumexp(Kt, dim=-1)
        v = epsilon * v_ + v
        # Size of the change we have performed on u
        diff = torch.sum(torch.abs(u - u0), dim=-1) + torch.sum(torch.abs(v - v0), dim=-1)
        mean_diff = torch.mean(diff).detach()
        if mean_diff.item() < thresh:
            break
    # Transport plan pi = diag(a)*K*diag(b)
    K = log_boltzmann_kernel(cost, u, v, epsilon)
    gamma = torch.exp(K)
    # Sinkhorn distance
    loss = torch.sum(gamma * cost, dim=(-2, -1))
    return gamma, loss.mean()


def index_points(points, idx):
    """ Selects a subset of points from a larger set of points based on idx values.
    Input:
        feats: input feats data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed feats data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)    
    view_shape[1:] = [1] * (len(view_shape) - 1)    # (B, 1)
    repeat_shape = list(idx.shape)  
    repeat_shape[0] = 1 # (1, S)
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)  # (B,S)
    new_points = points[batch_indices, idx, :]
    return new_points


def gmm_params(gamma, pts, return_sigma=False):
    """ Return GMM parameters (pi, mu, sigma) given cluster assignments and points.
    Input:
        - gamma: [B,N,J=128]. Soft cluster assignments (i.e. transportation policy). 
        J is number of clusters, i.e. number of gaussian components in the GMM. Created by wkeans().
        - pts:   [B,N,D=3].
    Output: GMM parameters
        - pi:    [B,N]
        - mu:    [B,N,3]
        - sigma: [B,N,3,3](optional). Covariance matrices 
    """
    # pi: B feats J
    D = pts.size(-1)
    pi = gamma.mean(dim=1)              # (B,N,J) -> (B,J)
    npi = pi * gamma.shape[1] + 1e-5    # (B,J)
    # p: B feats J feats D
    mu = gamma.transpose(1, 2) @ pts / npi.unsqueeze(2)     # (B,J,N) @ (B,N,D) -> (B,J,D)
    if return_sigma:
        # diff: B feats N feats J feats D
        diff = pts.unsqueeze(2) - mu.unsqueeze(1)
        # sigma: B feats J feats 3 feats 3
        eye = torch.eye(D).unsqueeze(0).unsqueeze(1).to(gamma.device)
        sigma = (((diff.unsqueeze(3) @ diff.unsqueeze(4)).squeeze() *
                  gamma).sum(dim=1) / npi).unsqueeze(2).unsqueeze(3) * eye
        return pi, mu, sigma
    return pi, mu


def og_params(pts, gamma, o_score=None, feature=None):
    if o_score is not None:
        # score [B, N]
        gamma_ex = (1.0 - o_score)
        # score [B, N, 1]
        gamma_ex = gamma_ex.unsqueeze(-1)
        # score [B, N, J]
        score = torch.cat([torch.einsum('bnk,bn->bnk', gamma, o_score), gamma_ex], dim=-1)
    else:
        score = gamma
    # mu: B x J x 3
    pi, mu = gmm_params(score, pts)
    if feature is not None:
        fea_mu = gmm_params(score, feature)[1]
        return pi, mu, fea_mu
    return pi, mu


def farthest_point_sample(xyz, npoint, is_center=False):
    """
    Input:
        pts: pointcloud data, [B, N, 3]
        npoint: number of samples
        is_center: if True, the initial farthest point is selected as the centroid 
        of the entire point cloud xyz.
    Return:
        sub_xyz: sampled point cloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(xyz) * 1e10
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    if is_center:
        centroid = xyz.mean(1).view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    else:
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def wkeans(x, num_clusters, dst='feats', iters=10, is_fast=True):
    ''' Wasserstein Weighted K-means clustering 
    Inputs:
        - x: [B,N=717,C=3]. Point cloud.
        - num_clusters: int (J=128). Number of clusters.       
        - dst: str. Distance metric. 'feats' for feature space dist or 'eu' for
        euclidean distance.
    Important Intermediate Variables:
        - cost: [B,N,J]. Cost induced by x-to-centroids transportation policy.
        I.e. squared distance between x and centroids.
        - gamma: [B,N,J]. Transportation policy from x to cluster centroids.
        Also can be thought of as soft cluster assignments.
    Outputs:
    '''
    bs, num, dim = x.shape
    if is_fast:
        ids = farthest_point_sample(x, num_clusters, is_center=True)
        centroids = index_points(x, ids)    # (B,M,C=3)
    else:
        ids = torch.randperm(num)[:num_clusters]
        centroids = x[:, ids, :]            # (B,M,C=3)
    gamma, pi = torch.zeros((bs, num, num_clusters), requires_grad=True).to(x), None
    for i in range(iters):
        if dst == 'eu':
            cost = square_distance(x, centroids)
        else:
            x = F.normalize(x, p=2, dim=-1)
            centroids = F.normalize(centroids, p=2, dim=-1)
            cost = 2.0 - 2.0 * torch.einsum('bnd,bmd->bnm', x, centroids)
        gamma = num * sinkhorn(cost, max_iter=10)[0]
        pi, centroids = gmm_params(gamma, x)
    return gamma, pi, centroids


def cos_similarity(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    scores = torch.einsum('bnd,bmd->bnm', x, y)
    return scores


def cos_distance(x, y):
    return 2 - 2 * cos_similarity(x, y)


def contrastsk(x, y, p=None, epsilon=1e-3, thresh=1e-3, max_iter=30, dst='eu'):
    if dst == 'eu':
        cost = square_distance(x, y)
    else:
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        cost = 2.0 - 2.0 * torch.einsum('bnd,bmd->bnm', x, y)
    gamma, loss = sinkhorn(cost, None, p, epsilon, thresh, max_iter)
    return gamma, loss


def get_local_corrs(xyz, xyz_mu, feats):
    """ Get anchor points in features space. Note that anchor points in euclidean
    space is points in xyz that are closest to xyz_mu.

    Inputs:
        - xyz:    [B,N,C=3]       Input point cloud 
        - xyz_mu: [B,J=128,3]     Anchor points in R^3. I.e. mu from GMM model.
        - feats:  [B,N,C'=520]    Input point features 
    
    Output:
        - feats_pos: [B,J=128,C'=520] Anchor points in feature space (R^520). 
    """
    dis = square_distance(xyz_mu, xyz)  # (B,J,N)
    # c_dims = xyz.size(-1)
    f_dims = feats.size(-1)                             # C'=520
    idx = torch.topk(dis, k=1, dim=2, largest=False)[1] # (B,J,1)
    idx = torch.nan_to_num(idx, nan=0)                  # (B,J,1)
    # xyz_idx = idx.repeat(1, 1, c_dims)
    # xyz_anchor = torch.gather(xyz, dim=1, index=xyz_idx)
    feats_idx = idx.repeat(1, 1, f_dims)# (B,J,C'=520)
    feats_pos = torch.gather(feats, dim=1, index=feats_idx)
    return feats_pos


def get_anchor_corrs(xyz, feats, num_clusters, dst='eu', iters=10, is_fast=True):
    '''
    Inputs:
        - xyz:          [B,C=3,N=717]    Input point cloud (channels first)
        - feats:        [B,C'=520,N=717] Additional input features for each point (channels first)
        - num_clusters: int. (J=128)     Number of clusters/anchor points to generate
        - dst: 'eu' for Euclidean distance or 'feats' for feature space distance
    Outputs:
        - feats_anchor: [B,C',J=128]    Feature space anchor points
        - feats_pos:    [B,C',J=128]    Feature space cluster centroids
        - gamma:        [B,N, J=128]    R^3 xyz to centroids transportation policy
        - pi:           [B,J=128]       GMM param, Gaussian component weights
        - xyz_mu:       [B,C=3,J=128]   GMM param, centroids (channel first)
    '''
    gamma, pi, xyz_mu = wkeans(xyz.transpose(-1, -2), num_clusters, dst, iters, is_fast)
    feats_pos = gmm_params(gamma, feats.transpose(-1, -2))[1].transpose(-1, -2)
    feats_anchor = get_local_corrs(xyz.transpose(-1, -2), xyz_mu, feats.transpose(-1, -2)).transpose(-1, -2)
    return feats_anchor, feats_pos, gamma, pi, xyz_mu.transpose(-1, -2)
