import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.special import digamma, logsumexp

"""
参考:https://nbviewer.org/github/amber-kshz/PRML/blob/master/notebooks/Ch10_Variational_Inference_for_Gaussian_Mixture_Model.ipynb
"""

path = ["216","219"]
p=1
def gaussian_ellipse_points(mean, cov, n_std=2.0, num_points=100):

    #固有値分解
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    #楕円パラメータ
    theta = np.linspace(0, 2*np.pi, num_points)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)

    #楕円に変換
    ellipse_2d = n_std*circle@np.diag(np.sqrt(eigvals))@eigvecs.T
    ellipse_2d += mean

    ellipse_3d = np.zeros((num_points, 3))
    ellipse_3d[:, 0] = ellipse_2d[:, 0]
    ellipse_3d[:, 2] = ellipse_2d[:, 1]

    return ellipse_3d

def ellipse_lineset(points_3d, color=[1, 0, 0]):
 
    lines = [[i, (i+1) % len(points_3d)] for i in range(len(points_3d))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_3d)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color]*len(lines))

    return line_set


def mean_point_cloud(m_k):
    pts = np.zeros((m_k.shape[0], 3))
    pts[:, 0] = m_k[:, 0]
    pts[:, 2] = m_k[:, 1]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.paint_uniform_color([0, 1, 0])
    return pcd



pcd = o3d.io.read_point_cloud("1109_ver0_0_4_"+path[p]+"westimate_bo_x_z.pcd")
points_3d = np.asarray(pcd.points)
points_xz = points_3d[:, [0, 2]]
geoms = []
# geoms.append(pcd)
# o3d.visualization.draw_geometries(geoms)
Cluster = 20


N,D = points_xz.shape
K= Cluster
X=points_xz
seed =42

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
r /= r.sum(axis=1, keepdims=True)
print(r.shape)



for q in range(400):

    #m_step
    #分布の重みつきデータ数
    N_sum = r.sum(axis=0) 

    #分布重みつき平均
    x_ave = r.T @ X /  N_sum[:, None]


    S =  np.zeros((K, D, D))
    for i in  range(K):
        S_1 = np.cov(X, rowvar=False, aweights=r[:, i], bias=True)
        S[i] =  S_1

    alpha = alpha_0 + N_sum
    beta = beta_0 + N_sum
    nu = nu_0 + N_sum
    #nuの中心の調整
    m = (beta_0*m_0 + N_sum[:,None]*x_ave)/beta[:,None]

    #各ガウス分布の共分散行列の調整
    W_inv = np.zeros((K, D, D))
    for i in range(K):
        m_diff = (x_ave[i] - m_0)[:,None]
        #print((m_diff@m_diff.T).shape)
        W = W_0_inv+ N_sum[i]*S[i]+((beta_0 * N_sum[i])/(beta_0+N_sum[i]))*(m_diff@m_diff.T)
        W_inv[i] = np.linalg.inv(W)


    #Eステップ　rを求めたい
    pho = []
    pho_1=  np.zeros((N, K))
    pho_2 = np.zeros(K)
    for i in range(K):
        diff = X - m[i]

        #これが遅すぎて無理!
        # pho1 = np.zeros(N)
        # for n in range(N):
        #     pho1[n] = diff[n] @ W_inv[i] @ diff[n]　

        # print("diff",diff.shape)
        # print("W_inv[i]",(W_inv[i]).shape)
        #diff(点群数，次元数)
        #W_inv[i](次元数,次元数)
        
        pho1 = np.einsum('nd,dd,nd->n', diff, W_inv[i], diff) #二次形式の公式

        pho = (D / beta[i]) + nu[i] * pho1

        pho_1[:, i] = -0.5*pho #1行目
        psi_sum = 0.0
        for h in range(1, D + 1):
            psi_sum += digamma((nu[i]+1.0-h)/2.0)
        logdat = np.log(np.linalg.det(W_inv[i])) 
        pho_2[i] =0.5*psi_sum + 0.5*logdat#2行目

    pho_3 = -0.5*(D*np.log(2.0)) #これがetaかな?
    pho_4 = digamma(alpha)-digamma(alpha.sum()) #3行目
    
    # print("pho_1", pho_1.shape)
    # print("pho_2", (pho_2[None, :]).shape)
    # print("pho_4", pho_4.shape)
    pho = pho_1+pho_2[None, :]+pho_3+pho_4[None, :]


    rho = np.exp(pho)
    r = rho / rho.sum(axis=1, keepdims=True)
    print("num",q)


#ノイズ判定
noise_filter =[]
for k in range(Cluster):
    Cov =  np.linalg.inv(W_inv[k]) / (nu[k] - D - 1)
    sigma_x = np.sqrt(Cov[0, 0])
    sigma_z = np.sqrt(Cov[1, 1])
    if (sigma_z / sigma_x) > 0.135:
        noise_filter.append(k)
        print(k)
        

is_removed = np.zeros(len(X), dtype=bool)

labels = np.argmax(r, axis=1)
for i in range(len(X)):
    k = labels[i]
    if k not in noise_filter:
        continue

    cov = np.linalg.inv(W_inv[k]) / (nu[k] - D - 1)
    inv_cov = np.linalg.inv(cov)

    diff = X[i] - m[k]
    mahal_sq = diff.T @ inv_cov @ diff

    if mahal_sq <= 2**2:
        is_removed[i] = True

points_removed = points_3d[is_removed]
points_remaining = points_3d[~is_removed]

geoms = []

#元の点群

pcd_save = o3d.geometry.PointCloud()
pcd_save.points = o3d.utility.Vector3dVector(points_remaining)
pcd_save.paint_uniform_color([0, 1, 1])  


pcd_noize = o3d.geometry.PointCloud()
pcd_noize.points = o3d.utility.Vector3dVector(points_removed)
pcd_noize.paint_uniform_color([1, 0, 0])

geoms.append(pcd_save)
geoms.append(pcd_noize)

#各クラスタの分布
for k in range(Cluster):
    ellipse_pts = gaussian_ellipse_points(
        mean=m[k],
        cov= np.linalg.inv(W_inv[k]) / (nu[k] - D - 1),
        n_std=2.0,
        num_points=100
    )
    ellipse = ellipse_lineset(ellipse_pts, color=[1, 0, 0])
    geoms.append(ellipse)

#平均点
geoms.append(mean_point_cloud(m))

#真値
pcd_true = o3d.io.read_point_cloud("1109_ver0_0_4_"+path[p]+"westimate_bo3.pcd")
pcd_trues = np.asarray(pcd_true.points)
pcd_true.paint_uniform_color([1, 0, 1])
geoms.append(pcd_true)


o3d.visualization.draw_geometries(geoms)
