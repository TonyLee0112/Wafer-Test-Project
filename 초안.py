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

#파일 불러오기
import os
print(os.listdir("../input"))

#경고창 무시하기
import warnings
warnings.filterwarnings("ignore")

#pickle 형식으로 저장된 데이터를 읽어들임
df=pd.read_pickle("../input/LSWMD.pkl") 
df.info()
df.head()
df.tail()

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
plt.show()

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
df.sample(5)

#분석 결과 632개의 wafermap 크기가 존재함.

# 이름 수정해주고, 불량의 유형을 0 ~ 8까지 정해줌.
df['failureNum']=df.failureType
df['trainTestNum']=df.trianTestLabel
mapping_type={'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,'Random':5,'Scratch':6,'Near-full':7,'none':8}

# 문자를 숫자로 바꿔서 쉽게 판단할라고 바꿔줌. 머신러닝 사전작업
mapping_traintest={'Training':0,'Test':1}
df=df.replace({'failureNum':mapping_type, 'trainTestNum':mapping_traintest})

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

#ax1 서브플롯에 파이 차트를 그리는 함수 / autopct = 조각의 비율로 백분율로 표시 / 소수점 한 자리까지 표시
ax1.pie(no_wafers, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)

uni_pattern=np.unique(df_withpattern.failureNum, return_counts=True)
labels2 = ['','Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']
ax2.bar(uni_pattern[0],uni_pattern[1]/df_withpattern.shape[0], color='gold', align='center', alpha=0.9)
# uni_pattern[0]: 실패 유형의 고유한 값, uni_pattern[1] / df_withpattern.shape[0]: 패턴이 있는 웨이퍼 중에 실패 유형의 비율 
ax2.set_title("failure type frequency")
ax2.set_ylabel("% of pattern wafers")
ax2.set_xticklabels(labels2)

plt.show()
# 결론: 실제 패턴이 있는 웨이퍼는 3.1%에 불과
# 첫 번째 시사점: 결함 패턴이 없으면 제외가능.
# 두 번재 서브플롯을 보면 Edge-Ring에서 결함이 제일 많이 발생함(35%이상). 그 뒤로 Edge-Loc, Center.
# Donut과 Near-full은 많이 발생하지 않는 수준
# 두 번째 시사점: Edge-Ring, Edge-Loc, Center 결함이 집중적으로 발생한다는 점에서, 반도체 제조 공정에서 이 결함들에 더 많은 주의를 기울여야 할 필요가 있음을 시사
# 머신러닝 돌릴 때도 이거에 초점을 더 맞출 필요가 있음

# 10개의 행을 가진 서브플롯 그리드 / 10개의 열을 가진 서브플롯 그리드
fig, ax = plt.subplots(nrows = 10, ncols = 10, figsize=(20, 20))

#2D를 1D로 변환
ax = ax.ravel(order='C')

# 이거 순서대로 0 ~ 99개를 추출한거야 그냥 순서대로 smapling을 한거
for i in range(100):
    img = df_withpattern.waferMap[i]
    ax[i].imshow(img)
    ax[i].set_title(df_withpattern.failureType[i][0][0], fontsize=10)
    ax[i].set_xlabel(df_withpattern.index[i], fontsize=8)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
plt.tight_layout()
plt.show() 

# 유형별로 10개 뽑기 총 80개
x = [0,1,2,3,4,5,6,7]
labels2 = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']

for k in x:
    fig, ax = plt.subplots(nrows = 1, ncols = 10, figsize=(18, 12))
    ax = ax.ravel(order='C')
    for j in [k]:
        img = df_withpattern.waferMap[df_withpattern.failureType==labels2[j]]
        for i in range(10):
            ax[i].imshow(img[img.index[i]])
            ax[i].set_title(df_withpattern.failureType[img.index[i]][0][0], fontsize=10)
            ax[i].set_xlabel(df_withpattern.index[img.index[i]], fontsize=10)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
    plt.tight_layout()
    plt.show() 

# 유형별로 1개 뽑기
x = [9,340, 3, 16, 0, 25, 84, 37]
labels2 = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']

#ind_def = {'Center': 9, 'Donut': 340, 'Edge-Loc': 3, 'Edge-Ring': 16, 'Loc': 0, 'Random': 25,  'Scratch': 84, 'Near-full': 37}
fig, ax = plt.subplots(nrows = 2, ncols = 4, figsize=(20, 10))
ax = ax.ravel(order='C')
for i in range(8):
    img = df_withpattern.waferMap[x[i]]
    ax[i].imshow(img)
    ax[i].set_title(df_withpattern.failureType[x[i]][0][0],fontsize=24)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
plt.tight_layout()
plt.show()

# 웨이퍼 맵을 13개의 영역으로 나누어 불량 유형에 따라 결함이 집중되는 위치를 분석하기 위한 전처리 과정
# 불량 유형에 따라 결함이 발생하는 위치가 다르기 때문에, 영역별 밀도 분포를 분석하기 위한 단계
# 13개란 중앙 사각형 9개 활꼴 4개
an = np.linspace(0, 2*np.pi, 100)
plt.plot(2.5*np.cos(an), 2.5*np.sin(an))
plt.axis('equal')
plt.axis([-4, 4, -4, 4])
plt.plot([-2.5, 2.5], [1.5, 1.5])
plt.plot([-2.5, 2.5], [0.5, 0.5 ])
plt.plot([-2.5, 2.5], [-0.5, -0.5 ])
plt.plot([-2.5, 2.5], [-1.5,-1.5 ])

plt.plot([0.5, 0.5], [-2.5, 2.5])
plt.plot([1.5, 1.5], [-2.5, 2.5])
plt.plot([-0.5, -0.5], [-2.5, 2.5])
plt.plot([-1.5, -1.5], [-2.5, 2.5])
plt.title(" Devide wafer map to 13 regions")
plt.xticks([])
plt.yticks([])
plt.show()


# 13개로 나뉜 각 영역의 결함 밀도를 계산하고, 이를 막대그래프로 시각화하여 결함 유형별 패턴을 분석
# 각 결함 유형이 웨이퍼에서 어느 영역에 발생하는지를 시각적으로 분석
def cal_den(x):
    return 100*(np.sum(x==2)/np.size(x))  

def find_regions(x):
    rows=np.size(x,axis=0)
    cols=np.size(x,axis=1)
    ind1=np.arange(0,rows,rows//5)
    ind2=np.arange(0,cols,cols//5)
    
    reg1=x[ind1[0]:ind1[1],:]
    reg3=x[ind1[4]:,:]
    reg4=x[:,ind2[0]:ind2[1]]
    reg2=x[:,ind2[4]:]

    reg5=x[ind1[1]:ind1[2],ind2[1]:ind2[2]]
    reg6=x[ind1[1]:ind1[2],ind2[2]:ind2[3]]
    reg7=x[ind1[1]:ind1[2],ind2[3]:ind2[4]]
    reg8=x[ind1[2]:ind1[3],ind2[1]:ind2[2]]
    reg9=x[ind1[2]:ind1[3],ind2[2]:ind2[3]]
    reg10=x[ind1[2]:ind1[3],ind2[3]:ind2[4]]
    reg11=x[ind1[3]:ind1[4],ind2[1]:ind2[2]]
    reg12=x[ind1[3]:ind1[4],ind2[2]:ind2[3]]
    reg13=x[ind1[3]:ind1[4],ind2[3]:ind2[4]]
    
    fea_reg_den = []
    fea_reg_den = [cal_den(reg1),cal_den(reg2),cal_den(reg3),cal_den(reg4),cal_den(reg5),cal_den(reg6),cal_den(reg7),cal_den(reg8),cal_den(reg9),cal_den(reg10),cal_den(reg11),cal_den(reg12),cal_den(reg13)]
    return fea_reg_den

df_withpattern['fea_reg']=df_withpattern.waferMap.apply(find_regions)

x = [9,340, 3, 16, 0, 25, 84, 37]
labels2 = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']

fig, ax = plt.subplots(nrows = 2, ncols = 4,figsize=(20, 10))
ax = ax.ravel(order='C')
for i in range(8):
    ax[i].bar(np.linspace(1,13,13),df_withpattern.fea_reg[x[i]])
    ax[i].set_title(df_withpattern.failureType[x[i]][0][0],fontsize=15)
    ax[i].set_xticks([])
    ax[i].set_yticks([])

plt.tight_layout()
plt.show()

# 이거는 라돈 변환을 한 이미지 값을 불러온건데 써야할지 말아야할지 모르겠어
def change_val(img):
    img[img==1] =0  
    return img

df_withpattern_copy = df_withpattern.copy()
df_withpattern_copy['new_waferMap'] =df_withpattern_copy.waferMap.apply(change_val)

x = [9,340, 3, 16, 0, 25, 84, 37]
labels2 = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']

fig, ax = plt.subplots(nrows = 2, ncols = 4, figsize=(20, 10))
ax = ax.ravel(order='C')
for i in range(8):
    img = df_withpattern_copy.waferMap[x[i]]
    theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    sinogram = radon(img, theta=theta)    
      
    ax[i].imshow(sinogram, cmap=plt.cm.Greys_r, extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')
    ax[i].set_title(df_withpattern_copy.failureType[x[i]][0][0],fontsize=15)
    ax[i].set_xticks([])
plt.tight_layout()

plt.show() 

# 웨이퍼 결함 패턴의 통계적 특징을 분석하고, 라돈 변환을 사용해 이미지의 평균과 표준편차를 구한 후,
# Cubic 보간법을 적용해 데이터를 시각화 가능하도록 변환하는 함수들입니다.

# 웨이퍼 결함 이미지의 라돈 변환을 이용해 평균 값을 계산한 후 Cubic 보간법을 적용하는 함수
def cubic_inter_mean(img):
    theta = np.linspace(0., 180., max(img.shape), endpoint=False)  # 0도부터 180도까지 라돈 변환 각도 설정
    sinogram = radon(img, theta=theta)  # 웨이퍼 이미지에 라돈 변환 적용
    xMean_Row = np.mean(sinogram, axis=1)  # 각 라돈 변환 결과의 행별 평균 계산
    x = np.linspace(1, xMean_Row.size, xMean_Row.size)  # x축 좌표 생성
    y = xMean_Row  # y값은 라돈 변환된 이미지의 평균값
    f = interpolate.interp1d(x, y, kind='cubic')  # Cubic 보간법 적용
    xnew = np.linspace(1, xMean_Row.size, 20)  # 보간된 새로운 x축 좌표 설정
    ynew = f(xnew) / 100  # 보간된 y축 값 반환, 100으로 나누어 스케일 조정
    return ynew  # 최종 변환된 데이터 반환

# 웨이퍼 결함 이미지의 라돈 변환을 이용해 표준편차를 계산한 후 Cubic 보간법을 적용하는 함수
def cubic_inter_std(img):
    theta = np.linspace(0., 180., max(img.shape), endpoint=False)  # 라돈 변환 각도 설정
    sinogram = radon(img, theta=theta)  # 웨이퍼 이미지에 라돈 변환 적용
    xStd_Row = np.std(sinogram, axis=1)  # 각 라돈 변환 결과의 행별 표준편차 계산
    x = np.linspace(1, xStd_Row.size, xStd_Row.size)  # x축 좌표 생성
    y = xStd_Row  # y값은 라돈 변환된 이미지의 표준편차 값
    f = interpolate.interp1d(x, y, kind='cubic')  # Cubic 보간법 적용
    xnew = np.linspace(1, xStd_Row.size, 20)  # 보간된 새로운 x축 좌표 설정
    ynew = f(xnew) / 100  # 보간된 y축 값 반환, 100으로 나누어 스케일 조정
    return ynew  # 최종 변환된 데이터 반환

# 각 웨이퍼 이미지에 대해 cubic_inter_mean 함수 적용, 결과를 'fea_cub_mean' 컬럼에 저장
df_withpattern_copy['fea_cub_mean'] = df_withpattern_copy.waferMap.apply(cubic_inter_mean)

# 각 웨이퍼 이미지에 대해 cubic_inter_std 함수 적용, 결과를 'fea_cub_std' 컬럼에 저장
df_withpattern_copy['fea_cub_std'] = df_withpattern_copy.waferMap.apply(cubic_inter_std)

# fea_cub_mean 컬럼의 처음 5개 데이터 확인
print(df_withpattern_copy['fea_cub_mean'].head())

# 첫 번째 웨이퍼 이미지의 fea_cub_mean 배열 데이터 확인
print(df_withpattern_copy['fea_cub_mean'][0])


# 결함 유형별 fea_cub_mean 값을 스케일링하여 바 그래프를 시각화하는 코드
x = [9, 340, 3, 16, 0, 25, 84, 37]  # 시각화할 결함 유형의 인덱스 목록
labels2 = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']  # 결함 유형 라벨

# 서브플롯 생성 (2행 4열)
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
ax = ax.ravel(order='C')  # 2D 배열을 1D 배열로 변환하여 각 서브플롯에 쉽게 접근

# 데이터 스케일링을 위한 값 설정
scaling_factor = 10000  # Cubic 보간 평균 값을 10000배 확대

# 다시 서브플롯 생성
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
ax = ax.ravel(order='C')

# 각 결함 유형별로 그래프 그리기
for i in range(8):  # 8개의 결함 유형에 대해 반복
    if x[i] < len(df_withpattern_copy):  # 유효한 인덱스인지 확인
        data = df_withpattern_copy['fea_cub_mean'][x[i]] * scaling_factor  # fea_cub_mean 값을 스케일링
        if isinstance(data, np.ndarray):  # 데이터가 배열인지 확인
            ax[i].bar(np.linspace(1, len(data), len(data)), data)  # 바 그래프 그리기
            ax[i].set_title(df_withpattern_copy['failureType'][x[i]][0][0], fontsize=10)  # 각 결함 유형에 대한 제목 설정
            ax[i].set_xticks([])  # x축 눈금 제거
            ax[i].set_xlim([0, len(data) + 1])  # x축 범위 설정
            ax[i].set_ylim([0, np.max(data) + 0.1])  # y축 범위를 스케일링된 데이터에 맞게 설정
        else:
            print(f"Data at index {x[i]} is not an array")  # 배열이 아닌 경우 경고 출력
    else:
        print(f"Index {x[i]} is out of bounds")  # 인덱스가 범위를 벗어난 경우 경고 출력

plt.tight_layout()  # 플롯 간의 간격을 자동으로 조정
plt.show()  # 그래프를 화면에 출력



# 웨이퍼 결함 맵의 가장 두드러진 영역(salient region)을 식별하여 해당 영역을 시각화
# 결함 패턴에서 노이즈를 제거하고 주요 결함 영역을 추출하는 과정
# 이미지 처리 기법인 영역 레이블링(region labeling)을 사용하여 가장 큰 영역을 식별하고, 그 영역을 시각화
# 가장 두드러진 영역을 식별하는 것은 결국 노이즈를 필터링 하는 것과 같다.
# 따라서, 우리는 가장 두드러지는 부분의 최대 영역을 고르고 region labeling 알고리즘을 사용할 것이다.
# 이러한 두드러지는 영역에서 면적, 둘레, 장축길이, 단축길이, 견고함, 이심률 등등의 기하학적 특성을 뽑아내볼 것이다.

x = [9, 340, 3, 16, 0, 25, 84, 37]  # 시각화할 결함 유형의 인덱스
labels2 = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']  # 결함 유형 라벨

# 서브플롯을 2행 4열로 생성하여 8개의 결함 패턴을 시각화
fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
ax = ax.ravel(order='C')  # 2D 배열을 1D 배열로 변환하여 서브플롯을 쉽게 접근할 수 있도록 변경

# 각 결함 패턴에 대해 시각화
for i in range(8):
    img = df_withpattern_copy.waferMap[x[i]]  # 결함 유형별 웨이퍼 맵 이미지 가져오기
    zero_img = np.zeros(img.shape)  # 0으로 채워진 동일한 크기의 빈 이미지 생성
    
    # 이미지 레이블링: 동일한 연결된 픽셀에 동일한 레이블을 부여하여 각 영역을 식별
    img_labels = measure.label(img, connectivity=1, background=0)  # 연결성(connectivity)을 고려하여 레이블링
    img_labels = img_labels - 1  # 레이블을 0부터 시작하도록 조정 (배경은 -1 처리)
    
    if img_labels.max() == 0:  # 만약 결함 영역이 없는 경우
        no_region = 0  # 기본 영역을 0으로 설정
    else:
        # 레이블 중 가장 빈도가 높은 영역(즉, 가장 큰 영역)을 식별
        info_region = stats.mode(img_labels[img_labels > -1], axis=None)
        no_region = info_region[0]  # 가장 큰 영역의 레이블 값을 추출
    
    # 해당 영역을 이미지에서 표시 (가장 큰 영역에 값을 할당하여 시각화)
    zero_img[np.where(img_labels == no_region)] = 2
    ax[i].imshow(zero_img)  # 이미지 출력
    ax[i].set_title(df_withpattern_copy.failureType[x[i]][0][0], fontsize=10)  # 각 결함 유형의 제목 설정
    ax[i].set_xticks([])  # x축 눈금 제거

# 레이아웃을 자동으로 조정하여 서브플롯 간의 간격을 균등하게 배치
plt.tight_layout()
plt.show()


# 두 점 사이의 유클리드 거리 계산 함수
def cal_dist(img, x, y):
    dim0 = np.size(img, axis=0)  # 이미지의 행 크기
    dim1 = np.size(img, axis=1)  # 이미지의 열 크기
    dist = np.sqrt((x - dim0 / 2) ** 2 + (y - dim1 / 2) ** 2)  # 중심점과의 거리 계산
    return dist

# 웨이퍼 맵의 기하학적 특성을 추출하는 함수
def fea_geom(img):
    norm_area = img.shape[0] * img.shape[1]  # 이미지 전체 영역
    norm_perimeter = np.sqrt((img.shape[0]) ** 2 + (img.shape[1]) ** 2)  # 이미지 대각선 길이(둘레 정규화)

    # 레이블링을 통해 각 영역을 식별
    img_labels = measure.label(img, connectivity=1, background=0)

    # 결함 영역이 없는 경우 처리
    if img_labels.max() == 0:
        img_labels[img_labels == 0] = 1  # 결함이 없으면 배경을 1로 설정
        no_region = 0  # 첫 번째 영역만 사용
    else:
        # 가장 빈번하게 나타나는 영역을 식별 (가장 큰 영역)
        info_region = stats.mode(img_labels[img_labels > 0], axis=None)
        
        # 배열 형식의 info_region.mode 처리
        if isinstance(info_region.mode, np.ndarray) and len(info_region.mode) > 0:
            no_region = info_region.mode[0] - 1  # 가장 빈번한 영역의 인덱스 계산
        else:
            no_region = info_region.mode - 1  # 배열이 아니면 직접 사용

    # 영역의 속성 정보 추출
    prop = measure.regionprops(img_labels)

    # 유효한 영역인지 확인
    if len(prop) == 0 or no_region >= len(prop) or no_region < 0:
        raise IndexError(f"Invalid region index: {no_region}, total regions: {len(prop)}")

    # 각 속성값 계산 (면적, 둘레, 중심 거리 등)
    prop_area = prop[no_region].area / norm_area  # 면적 정규화
    prop_perimeter = prop[no_region].perimeter / norm_perimeter  # 둘레 정규화
    prop_cent = prop[no_region].local_centroid  # 영역의 중심
    prop_cent_distance = cal_dist(img, prop_cent[0], prop_cent[1])  # 중심점과 이미지 중심과의 거리 계산
    prop_majaxis = prop[no_region].major_axis_length / norm_perimeter  # 주요 축 길이 정규화
    prop_minaxis = prop[no_region].minor_axis_length / norm_perimeter  # 부 축 길이 정규화
    prop_ecc = prop[no_region].eccentricity  # 이심률
    prop_solidity = prop[no_region].solidity  # 고형성

    # 기하학적 특성값 반환
    return prop_area, prop_perimeter, prop_majaxis, prop_minaxis, prop_ecc, prop_solidity, prop_cent_distance

# 기하학적 특성 계산 함수 적용 (각 웨이퍼 맵에 대해 기하학적 특성 추출)
df_withpattern_copy['fea_geom'] = df_withpattern_copy['waferMap'].apply(fea_geom)

# 특정 웨이퍼 맵에 대해 추출된 기하학적 특성 확인 (Donut 패턴 예시)
df_withpattern_copy.fea_geom[340] #Donut

# 밀도기반 : 13개
# 라돈기반 : 40개
# 기하기반 : 6개
# 전체 59개




# 전체 데이터프레임 복사
df_all = df_withpattern_copy.copy()

# fea_reg (13개 특징), fea_cub_mean (20개 특징), fea_cub_std (20개 특징), fea_geom (6개 특징) 추출
a = [df_all.fea_reg[i] for i in range(df_all.shape[0])]  # 13개 특징 (지역별 밀도)
b = [df_all.fea_cub_mean[i] for i in range(df_all.shape[0])]  # 20개 특징 (cubic interpolation mean)
c = [df_all.fea_cub_std[i] for i in range(df_all.shape[0])]  # 20개 특징 (cubic interpolation std)
d = [df_all.fea_geom[i] for i in range(df_all.shape[0])]  # 6개 기하학적 특성

# 59개의 특징 벡터로 결합
fea_all = np.concatenate((np.array(a), np.array(b), np.array(c), np.array(d)), axis=1)  # 총 59개의 특성

# 라벨 추출
label = [df_all.failureNum[i] for i in range(df_all.shape[0])]  # 라벨(결함 유형 번호) 추출
label = np.array(label)  # 배열로 변환


# 특징 벡터(X)와 라벨(y) 설정
X = fea_all  # 59개의 특징으로 구성된 특징 벡터
y = label  # 각 웨이퍼 맵의 결함 유형을 나타내는 라벨

# 학습 데이터와 테스트 데이터를 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print('Training target statistics: {}'.format(Counter(y_train)))  # 학습 데이터의 클래스 분포 확인
print('Testing target statistics: {}'.format(Counter(y_test)))  # 테스트 데이터의 클래스 분포 확인

# 클래스 레이블을 One-hot 인코딩으로 변환 (다중 클래스 분류를 위해)
y_train = to_categorical(y_train)  # 학습 데이터 라벨을 one-hot 인코딩
y_test = to_categorical(y_test)  # 테스트 데이터 라벨을 one-hot 인코딩

RANDOM_STATE = 42  # 난수 생성 시드를 고정하여 결과 재현성을 보장

# One-hot 인코딩된 라벨을 다시 1D 배열로 변환 (각 클래스에 대한 인덱스로 변환)
y_train_1d = np.argmax(y_train, axis=1)  # 각 클래스의 인덱스를 반환하여 1D 라벨로 변환
y_test_1d = np.argmax(y_test, axis=1)  # 테스트 데이터도 동일하게 처리

# One-Vs-One 다중 클래스 분류기를 사용하여 학습
# 각 클래스 쌍에 대해 이진 분류기를 학습하고 예측
clf2 = OneVsOneClassifier(LinearSVC(random_state=RANDOM_STATE)).fit(X_train, y_train_1d)

# 학습 데이터와 테스트 데이터에 대한 예측
y_train_pred = clf2.predict(X_train)
y_test_pred = clf2.predict(X_test)

# 학습 정확도 계산 (예측값이 실제값과 같은 비율)
train_acc2 = np.sum(y_train_1d == y_train_pred, axis=0, dtype='float') / X_train.shape[0]
# 테스트 정확도 계산
test_acc2 = np.sum(y_test_1d == y_test_pred, axis=0, dtype='float') / X_test.shape[0]

# 학습 및 테스트 정확도 출력
print('One-Vs-One Training acc: {}'.format(train_acc2 * 100))  # 학습 데이터에 대한 정확도
print('One-Vs-One Testing acc: {}'.format(test_acc2 * 100))  # 테스트 데이터에 대한 정확도
print("y_train_pred[:100]: ", y_train_pred[:100])  # 학습 데이터 예측 상위 100개 출력
print("y_train[:100]: ", y_train_1d[:100])  # 실제 학습 데이터 라벨 상위 100개 출력



def plot_confusion_matrix(cm, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    이 함수는 주어진 혼동 행렬(confusion matrix)을 시각화합니다.
    normalize=True로 설정하면 정규화된 혼동 행렬을 그립니다.
    
    Parameters:
    cm : 혼동 행렬
    normalize : 정규화 여부 (기본값: False)
    title : 제목 (기본값: 'Confusion matrix')
    cmap : 색상 매핑 (기본값: Blues 컬러맵 사용)
    """
    if normalize:
        # 정규화된 혼동 행렬로 변환
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # 혼동 행렬을 이미지로 표시
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    # 혼동 행렬 위에 숫자 출력 설정 (정규화 여부에 따라 포맷 지정)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # 혼동 행렬의 레이아웃 설정
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# y_test를 1D 배열로 변환 (One-hot 인코딩을 해제)
y_test_1d = np.argmax(y_test, axis=1)

# 혼동 행렬 계산 (예측값과 실제값을 비교하여 생성)
cnf_matrix = confusion_matrix(y_test_1d, y_test_pred)
np.set_printoptions(precision=2)  # 출력 정밀도 설정

# 혼동 행렬을 시각화하는 함수 정의 (정규화 유무 선택 가능)
def plot_confusion_matrix(cm, normalize=False, title='Confusion matrix'):
    """
    이 함수는 주어진 혼동 행렬을 시각화하며, 필요 시 정규화를 수행합니다.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 행별로 정규화

    # 혼동 행렬을 이미지로 표시
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    # 각 셀에 값 표시
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], '.2f' if normalize else 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    # 레이아웃 설정
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# 두 개의 서브플롯 설정: 정규화 및 비정규화된 혼동 행렬
fig = plt.figure(figsize=(15, 8))  # 전체 그래프 크기 설정
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])  # 2개의 서브플롯 생성

# 첫 번째 서브플롯: 비정규화된 혼동 행렬
plt.subplot(gs[0])
plot_confusion_matrix(cnf_matrix, title='Confusion matrix')

# 두 번째 서브플롯: 정규화된 혼동 행렬
plt.subplot(gs[1])
plot_confusion_matrix(cnf_matrix, normalize=True, title='Normalized confusion matrix')

# 전체 그래프 출력
plt.show()
