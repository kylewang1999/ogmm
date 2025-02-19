## Overlap-guided Gaussian Mixture Models for Point Cloud Registration

### Intorduction
Probabilistic 3D point cloud registration methods have shown competitive performance in overcoming noise, outliers, and density variations. 
However, registering point cloud pairs in the case of partial overlap is still a challenge. 
This paper proposes a novel overlap-guided probabilistic registration approach that computes the optimal transformation from matched Gaussian Mixture Model (GMM) parameters.
We reformulate the registration problem as the problem of aligning two Gaussian mixtures such that a statistical discrepancy measure between the two corresponding mixtures is minimized. 
We introduce a Transformer-based detection module to detect overlapping regions, and represent the input point clouds using GMMs by guiding their alignment through overlap scores computed by this detection module.
Experiments show that our method achieves superior registration accuracy and efficiency than state-of-the-art methods when handling point clouds with partial overlap and different densities on synthetic and real-world datasets.
Now it is a draft version, we will update it as soon as possible.
If you find our work useful, please consider citing our paper:
```
@inproceedings{mei2022overlap,
  title={Overlap-guided Gaussian Mixture Models for Point Cloud Registration},
  author={Mei, Guofeng and Poiesi, Fabio and Saltori, Cristiano and Zhang, Jian and Ricci, Elisa and Sebe, Nicu},
  booktitle={IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2023},
}
```
### Usage
#### Prerequisite
Our model is trained with the following environment:

* Python 3.8.8
* PyTorch 1.9.1 with torchvision 0.10.1 (Cuda 11.1)
* coloredlogs
* easydict
* h5py
* GitPython
* nibabel
* numpy
* scipy
* open3d
* tensorboard

#### Build the env using conda
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge trimesh coloredlogs tqdm easydict gitpython nibabel tensorboard transforms3d plyfile
conda install -c anaconda h5py
```