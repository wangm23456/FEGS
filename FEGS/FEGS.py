import numpy as np
from math import pi, cos, sin
from itertools import product
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import eigs
from typing import Iterable, Tuple
import re


def coordinate():
    """
    坐标系
    return:
        P: 20种氨基酸矢量, shape=[20, 3]
        V: 400种氨基酸对的矢量:  shape=[20,20,3]
    """
    P = np.array([[cos(i*2*pi/20), sin(i*2*pi/20), 1] for i in range(1, 21)])
    V = np.zeros((20,20,3))
    for i,j in product(range(20), range(20)):
        V[i,j,:] = P[i, :] + (1/4) * (P[j, :]-P[i, :]) #1/4弦
    return P, V

def GRS(seq: str, P: np.array, V: np.array, M: np.array) -> np.array:
    """
    3D路径表示
    seq: 字符串
    P: 单氨基酸矢量, shape=[20,3]
    V: 双氨基酸矢量, shape=[20,20,3]
    M: 理化顺序序列
    不同的理化顺序中氨基酸在P,V具有不同的索引
    return:
        路径矩阵, shape=[158, len_seq, 3]
    """
    l_seq = len(seq)
    k = M.shape[0] # 158种理化
    # cha = 'ACDEFGHIKLMNPQRSTVWY'
    g = np.zeros((k, l_seq+1, 3)) #+1原点
    for j in range(k):
        c = np.zeros((l_seq+1,3))
        d = np.zeros((1,3)) # 存储双氨基酸频率的漂移项
        _ipre = -1
        # _ipre2 = -1
        for i in range(l_seq):
            p = np.array([0,0,1])
            _ix = M[j].find(seq[i])
            # _ix2 = cha.find(seq[i])
            if _ix >= 0:
                p = P[_ix, :]
                if i >= 1:
                    d = d*(i-1)/i
                if _ipre >= 0:
                    d = d + V[_ipre,_ix,:]/i
            c[i+1, :] = c[i, :] + p + d
            _ipre = _ix
            # _ipre2 = _ix2
        g[j,:,:] = c
    return g

def ME(W: np.array) -> float:
    """
    计算A的特征值
    A=欧氏距离矩阵./路径距离矩阵
    return:
        特征值
    """
    W = W[1:,:]
    x,y=W.shape
    D=pdist(W) 
    E=squareform(D) #两两距离
    sdist=np.zeros((x,x)) # 边长矩阵
    for i in range(x):
        for j in range(i, x):
            if j-i==1:
                sdist[i,j]=E[i,j]
            elif j-i>1:
                sdist[i,j]=sdist[i,j-1]+E[j-1,j]
    sdist=sdist+sdist.T+np.diag(np.ones((x)))
    L=E/sdist
    es,_ = eigs(L,k=1)
    es = np.real(es)[0]/x
    return  es

def SAD(seq: str, a: str) -> Tuple[Iterable[float], Iterable[float]]:
    """
    常规统计信息
    return:
        AAC: 单氨基酸频率 len=20
        DPC: 双氨基酸频率 len=400
    """
    len_seq=len(seq)
    len_a=len(a)
    if len_seq != 0:
        AAC = [len([m.start() for m in re.finditer(_a, seq)])/len_seq for _a in a]
        DPC = [len([m.start() for m in re.finditer(aj+ai, seq)])/(len_seq-1) for ai, aj in product(a, a)]
    else:
        AAC = [0 for _ in range(20)]
        DPC=[0 for _ in range(400)]
    # DPC = np.array([DPC]).reshape((20,20)).T.flatten().tolist()
    return AAC, DPC

def FEGS(seq: str) -> Iterable[Iterable[float]]:
    M = np.load("./M.npy")
    P,V = coordinate()
    char = "ARNDCQEGHILKMFPSTWYV"
    Vi = []
    grs = GRS(seq, P, V, M) #shape=[158, len(seq), 3]
    els = [ME(grs[u]) for u in range(grs.shape[0])]
    aac, dpc = SAD(seq, char)
    Vi.extend(els)
    Vi.extend(aac)
    Vi.extend(dpc)
    return Vi