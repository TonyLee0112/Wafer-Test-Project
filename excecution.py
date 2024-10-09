# 라이브러리 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from skimage import measure
from skimage.transform import radon, probabilistic_hough_line
from scipy import interpolate, stats
import tensorflow as tf  # Theano 대신 TensorFlow 사용
from sklearn.model_selection import train_test_split  # cross_validation 대신 최신 모듈 사용
from tensorflow.keras.utils import to_categorical  # np_utils의 to_categorical 대체
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import confusion_matrix
import itertools

#경고창 무시하기
import warnings
warnings.filterwarnings("ignore")

#pickle 형식으로 저장된 데이터를 읽어들임
df=pd.read_pickle("LSWMD.pkl")

# waferindex는 각 lot에서 wafer가 몇 번째인지를 나타냄. 한 lot은 보통 25개.
# 하지만 현실적으로 모든 lot이 완벽하게 25개가 아님. Why? 센서 오류, 프로세스 중단, wafer 손상 등등
# 이를 시각화한 이유는 모든 lot에 wafer가 25로 구성되어 있지 않음을 시각화하고, 결함이나 비정상적인 부분을 확인하기 위함임.
uni_Index=np.unique(df.waferIndex, return_counts=True)
plt.bar(uni_Index[0],uni_Index[1], color='gold', align='center', alpha=0.5)

#x축 위치 0, 위치에 대한 빈도는 1, gold색, 막대를 x축의 중앙에 정렬, 투명도는 0.5
plt.title(" wafer Index distribution")
plt.xlabel("index #")
plt.ylabel("frequency")
plt.xlim(0,26)
plt.ylim(30000,34000)

# 불량 모델을 분류하는데 있어서 waferindex는 중요한 변수가 아니고, 오히려 이거 때문에 오염될 수 있으니까 삭제.
# waferMapDim을 추가한 이유: wafermap의 크기가 어떻게 다른지를 알기 위함임. Why? wafer map의 크기에 따라서 수율이 달라짐. 불량도 달라지고.
# 보통 wafer는 8인치 12인치 이렇게 나뉘는데 그 안에서도 어떤 chip을 만드느냐에 따라서 픽셀로 die를 나타냈을 때 그 안에 픽셀 배열이 달라짐.

#열 또는 행을 제거 / axis=1 --> 열을 제거 / axis=0 --> 행을 제거 /
df = df.drop(['waferIndex'], axis = 1)
def find_dim(x):
    dim0=np.size(x,axis=0)
    dim1=np.size(x,axis=1)
    return dim0,dim1
df['waferMapDim']=df.waferMap.apply(find_dim)

#분석 결과 632개의 wafermap 크기가 존재함.

# 이름 수정해주고, 불량의 유형을 0 ~ 8까지 정해줌.
df['failureNum']=df.failureType
df['trainTestNum']=df.trianTestLabel
mapping_type={'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,'Random':5,'Scratch':6,'Near-full':7,'none':8}

# 문자를 숫자로 바꿔서 쉽게 판단할라고 바꿔줌. 머신러닝 사전작업
mapping_traintest={'Training':0,'Test':1}
df = df.replace({'failureNum':mapping_type, 'trainTestNum':mapping_traintest})

# 결함 라벨(failureNum)이 0에서 8 사이에 해당하는 웨이퍼들을 추출
df_withlabel = df[(df['failureNum']>=0) & (df['failureNum']<=8)]
df_withlabel =df_withlabel.reset_index()
# 결함 유형(failureNum)이 0에서 7까지인 웨이퍼들만 추출
df_withpattern = df[(df['failureNum']>=0) & (df['failureNum']<=7)]
df_withpattern = df_withpattern.reset_index()
# 결함 유형이 'none'(8번)인 웨이퍼들만 포함된 데이터셋. 이 웨이퍼들은 패턴 결함이 없는 것으로 간주.
df_nonpattern = df[(df['failureNum']==8)]

# 총 811,457개의 웨이퍼 데이터 중 172,950개의 웨이퍼는 결함 유형 有. 나머지는 결측치(날려야겠지?)
# 이 중 25,519개의 웨이퍼는 특정 결함 패턴(Center, Donut, Edge-Loc 등)을 가지고 있으며, 147,431개의 웨이퍼는 결함이 없음.

# 너비 20 폭 4.5 / 1행 2열 너비는 1 : 2.5 / 첫 번재 그리드 셀에 서브플롯을 형성 / 두 번째 그리드 셀에 서브플롯을 형성
# subplot: 하나의 figure 안에 여러 개의 개별 플롯을 배열할 수 있다
fig = plt.figure(figsize=(20, 4.5))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2.5])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])

tol_wafers = df.shape[0]
no_wafers=[tol_wafers-df_withlabel.shape[0], df_withpattern.shape[0], df_nonpattern.shape[0]]
#레이블이 없는 웨이퍼의 수 / 패턴이 있는 웨이퍼의 수 / 패턴이 없는 웨이퍼의 수
colors = ['silver', 'orange', 'gold']

# 첫번재 조각을 0.1만큼 밖으로 부풀림
explode = (0.1, 0, 0)  # explode 1st slice

# no-label: 라벨이 없는 웨이퍼 (즉, 결함 유형이 지정되지 않은 웨이퍼).
# label&pattern: 라벨이 있고 실제 결함 패턴이 있는 웨이퍼.
# label&non-pattern: 라벨이 있지만 결함 패턴이 없는 웨이퍼 (failureNum이 8인, 'none'으로 라벨된 웨이퍼).
labels = ['no-label','label&pattern','label&non-pattern']

uni_pattern=np.unique(df_withpattern.failureNum, return_counts=True)
labels2 = ['','Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']


# 13개로 나뉜 각 영역의 결함 밀도를 계산하고, 이를 막대그래프로 시각화하여 결함 유형별 패턴을 분석
# 각 결함 유형이 웨이퍼에서 어느 영역에 발생하는지를 시각적으로 분석
def cal_den(x):
    return 100 * (np.sum(x == 2) / np.size(x))


def find_regions(x):
    rows = np.size(x, axis=0)
    cols = np.size(x, axis=1)
    ind1 = np.arange(0, rows, rows // 5)
    ind2 = np.arange(0, cols, cols // 5)

    reg1 = x[ind1[0]:ind1[1], :]
    reg3 = x[ind1[4]:, :]
    reg4 = x[:, ind2[0]:ind2[1]]
    reg2 = x[:, ind2[4]:]

    reg5 = x[ind1[1]:ind1[2], ind2[1]:ind2[2]]
    reg6 = x[ind1[1]:ind1[2], ind2[2]:ind2[3]]
    reg7 = x[ind1[1]:ind1[2], ind2[3]:ind2[4]]
    reg8 = x[ind1[2]:ind1[3], ind2[1]:ind2[2]]
    reg9 = x[ind1[2]:ind1[3], ind2[2]:ind2[3]]
    reg10 = x[ind1[2]:ind1[3], ind2[3]:ind2[4]]
    reg11 = x[ind1[3]:ind1[4], ind2[1]:ind2[2]]
    reg12 = x[ind1[3]:ind1[4], ind2[2]:ind2[3]]
    reg13 = x[ind1[3]:ind1[4], ind2[3]:ind2[4]]

    fea_reg_den = []
    fea_reg_den = [cal_den(reg1), cal_den(reg2), cal_den(reg3), cal_den(reg4), cal_den(reg5), cal_den(reg6),
                   cal_den(reg7), cal_den(reg8), cal_den(reg9), cal_den(reg10), cal_den(reg11), cal_den(reg12),
                   cal_den(reg13)]
    return fea_reg_den


df_withpattern['fea_reg'] = df_withpattern.waferMap.apply(find_regions)

x = [9, 340, 3, 16, 0, 25, 84, 37]
labels2 = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
ax = ax.ravel(order='C')
for i in range(8):
    ax[i].bar(np.linspace(1, 13, 13), df_withpattern.fea_reg[x[i]])
    ax[i].set_title(df_withpattern.failureType[x[i]][0][0], fontsize=15)
    ax[i].set_xticks([])
    ax[i].set_yticks([])

plt.tight_layout()
plt.show()