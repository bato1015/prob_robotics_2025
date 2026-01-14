
## 確率ロボティクス　課題

## 実行環境と実行方法
### 実行環境
- Ubuntu 20.04 on WSL (Windows Subsystem for Linux)
- Open3D 0.13.0

### 実行方法
```
pip install open3d　#Open3Dがインストールされていない場合
git clone https://github.com/bato1015/prob_robotics_2025.git
cd prob_robotics_2025
python3 VariationalInferenceForBayesianGaussianMixturModel.py
```

---

  ### 変分推論を用いたノイズ除去システム
本課題では，以下の条件で取得した 2 次元点群データを用い，変分推論に基づくベイズ混合ガウスモデルによるノイズ除去システムを構築した．

・点群データは Microsoft 社製の Azure Kinect DK［1］を用いて計測し，計測結果からエッジ部分を抽出した点群を対象とした

・単視点からの計測ではオクルージョンが多く発生するため，奥行き方向である y 軸を除外し，x–z 平面上の 2 次元点群として処理を行った．

---


  ### 背景と目的
  Kinectからの計測データには，計測誤差やセンサノイズが含まれている．本課題で使用するエッジの点群データは距離ベースのクラスタリングを用いたエッジ間隔でパラメータ推定を行う．
  しかし距離ベースの手法は，ノイズの影響を受けやすく，前処理としてノイズ除去を行うことが望ましい．ノイズを含むデータに対して，EM法やk-means法は，分布の形状やクラスタリング数を事前に仮定する必要があるため適応が難しい．そこで本課題では，変分推論に基づくベイズ混合ガウスモデルを用い，ノイズ除去システムを構築する．
  
---


  ### 変分推論を用いたベイズ混合ガウスモデル
変分推論(Variational Inference)とは，解析的に計算できない未知の確率分布 $p(\boldsymbol{X}|D)$ を知るために，計算しやすい簡単な式を使って周辺化計算を近似的に実行する方法である．ここで, $\boldsymbol{X}$ は観測されてない未知の変数を示し, $D$ は観測データを示す．変分推論の他にもラプラス近似や期待値伝搬などがあげられる[2]．混合ガウス分布において未知のパラメータとなり決定したいのは,分布に関する事後分布 

$$p(\boldsymbol{\pi},\boldsymbol{\mu}_{1:K},\mathbf{\Lambda}_{1:K},k_{1:N}|\boldsymbol{x}_{1:N})$$  

と分布を決定する分布の事後分布

$$p(r_{ij},\boldsymbol{m}_j,\beta_{j},W_{j},\nu_{j},\alpha_j)$$

ここでiはデータの数($i= 0,1,2 \dots N$), j はガウス分布の数($ j= 0,1,2 ... K$)， pi はクラスタにデータがいる確率， nu_1:K は分布の平均値 Lambda_1:K は精度行列, k はクラスタ数， r_ij は i 番目のデータ x_i が j 番目のクラスタに所属する確率分布，m_j は分布の分布の中心，W_j , beta_j , nu_j は精度行列 Lambda_j を決定するウィシャート分布 の基準行列とそのパラメータ，alpha_j は pi_1:K を示す[4]．

---

### 変分推論のアルゴリズム[4]
変分推論の解法の一つにEM法に似たアプローチで分布の決定を行う手法がある．
これはパラメータ群同士が独立していると仮定し，以下の式のように2つの分布の積で構成される近似の分布qを作成する．

$$q(\pi_{1:K},\boldsymbol{\mu}_{1:K}, \Lambda_{1:K}, k_{1:N}) = q_1(k_{1:N})q_2(\pi_{1:K},\boldsymbol{\mu}_{1:K}, \Lambda_{1:K})$$

この式を用いて，一つのパラメータ群を固定し，片一方を計算という処理を繰り返してパラメータを決定する．
Mステップのように分布を決定するにはq_1を固定し，q_2を動かし，Eステップのように分布の信頼度を決定するにはq_2を固定し，q_1を動かす．
下記に各ステップの具体的なアルゴリズムを示す．

#### 初期値を決定
#### Mstep
- 以下の式に従って事後分布m_1:K, beta_1:K, W_1:K, nu_1:K, alpha_1:Kを決定する．以下の数式の解説は[3]参照

  $$N_j = \sum_{i=1}^N r_{ij}\qquad\qquad$$

  $$\bar{\boldsymbol{x}}j = \dfrac{1}{N_j} \sum_{i=1}^N r_{ij}\boldsymbol{x}_i\\quad$$
  
  $$\Sigma_j = \dfrac{1}{N_j} \sum_{i=1}^N r_{ij}(\boldsymbol{x}_i - \bar{\boldsymbol{x}}_j)(\boldsymbol{x}_i - \bar{\boldsymbol{x}}_j)^\top$$
  
  $$\alpha_j, \beta_j, \nu_j) =(\alpha_j', \beta_j', \nu_j') + (N_j, N_j, N_j)$$
  
  $$\boldsymbol{m}_j = (\beta_j' \boldsymbol{m}_j' + N_j \bar{\boldsymbol{x}}_j ) /\beta_j\qquad\qquad\qquad$$
  
  $$W^{-1}_j = W'^{-1}_j + N_j \Sigma_j + \dfrac{\beta'_j N_j}{\beta'_j+ N_j} (\bar{\boldsymbol{x}}_j - \boldsymbol{m}_j')(\bar{\boldsymbol{x}}_j - \boldsymbol{m}_j')^\top$$


#### Estep
- 以下の式に従ってr_ijを決定する．以下の数式の解説は[3]参照
 E_stepの1行目のマハラノビス距離^2である（ x_i - m_j ） ^ T W_j（ x_i - m_j ）は2次形式よりΣ^n (（ x_i - m_j ）_n・ W_inv・（ x_i - m_j ）_n)が導出できるため，実装にはこれを利用した．

$$r_{ij} = \eta \rho_{ij}$$
  
$$\log_e \rho_{ij} = -\dfrac{1}{2} d \beta_j^{-1} -\dfrac{1}{2} \nu_j(\boldsymbol{x}_i - \boldsymbol{m}_j)^\top W_j (\boldsymbol{x}_i - \boldsymbol{m}_j)$$
         
$$\quad+ \dfrac{1}{2} \sum_{h=1}^d \psi\left(\dfrac{\nu_j + 1 - h}{2}\right) + \dfrac{1}{2}\log_e | W_j | + \eta'$$
        
$$\quad+\psi(\alpha_j) - \psi\left( \sum_{j=1}^K \alpha_j \right)$$


決定したr_ijを用いてMstepへ

---

### 対応するパラメータ
以下が使用したパラメータである．本講義では，2次元の点群データを対象としているためD=2となる．Nは点群数に対応し，Kはノイズとエッジのクラスタリング数を示す．説明は[5]のサイトを参考にした．
| Variable name | Description | Shape |
|--------------|------------|-------|
| `X` | points | `(N, D)` |
| `r` | Responsibility  | `(N, K)` |
| `N_sum` |Weighted count of data| `(K,)` |
| `x_ave` |  Weighted responsibility mean of data | `(K, D)` |
| `S` | weighted responsibility covariance matrices | `(K, D, D)` |
| `alpha_0` | Dirichlet prior parameter | scalar |
| `beta_0` | Prior precision of mean | scalar |
| `m_0` | Prior mean vector | `(D,)` |
| `nu_0` | Prior wishart parameter| scalar |
| `W_0` | Prior wishart parameter| `(D, D)` |
| `W_0_inv` | Inverse of prior precision matrix | `(D, D)` |
| `alpha` | Posterior Dirichlet parameter | `(K,)` |
| `beta` | Posterior precision of mean | `(K,)` |
| `m` | Posterior mean vectors | `(K, D)` |
| `nu` | Posterior wishart parameter | `(K,)` |
| `W_inv` | Posterior wishart parameter | `(K, D, D)` |


---


## ノイズの判定式
本課題では，変分推論によって求めたパラメータを用いて，2段階の判定により点群中のノイズを判定する．
### ノイズクラスタの判定
各クラスタに対して，共分散行列を次式で求め，Sigmaから標準偏差を計算する．

`Sigma` =  `W_0_inv` ^(-1) / (nu - D - 1)

今回のエッジの点群データは正確な計測を行えた場合，x方向に広く，z方向に狭い点群が複数できる．この特性を利用し，z方向とx方向の広がりの比が閾値tを超えるクラスタをノイズまたはノイズを含むエッジのクラスタと判定する．ここで，`sigma_x_k`,`sigma_z_k`はクラスタkのx,z方向の標準偏差を示す．

`sigma_z_k` / `sigma_x_k` > ta

### ノイズ点の判定
ノイズクラスタと判定されたクラスタを対象にマハラノビス距離を計算し，閾値以下の点をノイズ点と設定する．
点iは上記の2つの判定を同時に満たす場合，ノイズ点として除去される．

---

## 結果

本課題では，ノイズ少ないデータ（p=0）とノイズが多いデータ(p=1)の2つを用意し，検証を行い，結果は以下のようになった．

画像内の赤色の点群がノイズ点と判定された点，青色が残った点群，緑色の点が変分推論により求めたmの座標，ピンク色の点が実際に計測した計測点を示す．
特に図2ののようなノイズが多いデータでは，距離ベースのクラスタリングで邪魔となるノイズを削除できたことがわかった．

<img src="https://github.com/user-attachments/assets/06a5c5ec-a46e-419f-a11a-80583069812d" width=400>

図1 ノイズが少ないデータ

<img src="https://github.com/user-attachments/assets/5eb7b550-7924-4499-a887-a50355ba22fc" width=400>

図2 ノイズが多いデータ

---


## LLMの利用について
ノイズ除去の判定法・描画・初期値のパラメータ設定にChat GPTまたは，Geminiを利用した．


---



  ### 参考文献
  以下に参考にしたサイトおよび書籍を記載する．主に，書籍[3],講義資料[4],コード[6]を参考に実装を行った．
  
[1] Microsoft,"Azure Kinect DK",(URL:https://azure.microsoft.com/ja-jp/products/kinect-dk/?msockid=286607017653682f2561121677ca69fe).accessed:2026/01/13

[2]須山　敦志ら,"ベイズ推論による機械学習入門"，講談社，2017.

[3]上田 隆一,"ロボットの確率・統計～製作・競技・知能研究で役立つ考え方と計算法"，コロナ社，2024

[4]ryuichiueda,"確率ロボティクス第8回: 機械学習（その2）",github,(URL:https://ryuichiueda.github.io/slides_marp/prob_robotics_2025/lesson8-2.html),accessed:2026/01/14

[5]zuka,"【徹底解説】変分ベイズをはじめからていねいに",Academaid,(URL:https://academ-aid.com/ml/vb),accessed:2026/01/14

[6]amber-kshz,"Variational inference for Bayesian Gaussian mixture models",(URL:https://github.com/amber-kshz/PRML),accessed:2026/01/14
