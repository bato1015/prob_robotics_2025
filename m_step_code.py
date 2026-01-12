import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.special import digamma, logsumexp

"""
参考:https://nbviewer.org/github/amber-kshz/PRML/blob/master/notebooks/Ch10_Variational_Inference_for_Gaussian_Mixture_Model.ipynb
"""

pcd = o3d.io.read_point_cloud("1109_ver0_0_4_216westimate_bo_x_z.pcd")
points_3d = np.asarray(pcd.points)
points_xz = points_3d[:, [0, 2]]
geoms = []
# geoms.append(pcd)
# o3d.visualization.draw_geometries(geoms)
Cluster = 100

#m_step
N,D = points_xz.shape
K= Cluster
X=points_xz
seed =100
epsilon = 1e-12

alpha_0 = 0.003 #事前分布の信頼度(学習を進めると高まるため、初期は小さく)
beta_0 = 0.9 #事前分布の精度を決定するパラメータbeta>0 https://en.wikipedia.org/wiki/Normal-Wishart_distribution?utm_source=chatgpt.com
m_0 = np.mean(X, axis=0)#分布の中心
nu_0 = D + 20.0 #Wishart分布のパラメータ
sigma_x = 0.25 #精度行列の事前のばらつき
sigma_z = 0.01

W_0 = np.diag([
    1.0 / sigma_x**2,
    1.0 / sigma_z**2
])
W_0_inv = np.linalg.inv(W_0)


rng = np.random.default_rng(seed)
r = rng.random((N, Cluster))
print(r.shape)
#分布の重みつきデータ数
N_sum = r.sum(axis=0) + epsilon

#分布重みつき平均
x_ave = r.T @ X /  N_sum[:, None]

S =  np.zeros((K, D, D))
for i in  range(K):
    S[i] = np.cov(X,rowvar=False,aweights=r[:,i],bias=True)


alpha = alpha_0 + N_sum
beta = beta_0 + N_sum
nu = nu_0 + N_sum
#nuの中心の調整
m = (beta_0*m_0 + N_sum[:,None]*x_ave)/beta[:,None]
#各ガウス分布の共分散行列の調整
W_inv = np.zeros((K, D, D))
for i in range(K):
    m_diff = (x_ave[i] - m_0)
    W = W_0_inv+ N_sum[i]*S[i]+((beta_0 * N_sum[i])/(beta_0+N_sum[i]))\
    *(m_diff@m_diff.T)
    W_inv[i] = np.linalg.inv(W)
print(W_inv)