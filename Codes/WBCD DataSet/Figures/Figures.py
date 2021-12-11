import numpy as np
# from ttictoc import tic,toc
import matplotlib.pyplot as plt
import pylab
from numpy import linalg as LA
# import tensorflow as tf
import math
import scipy.io

params = {'backend': 'ps',
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'legend.handlelength': 1,
          'legend.borderaxespad': 0,
          'font.family': 'serif',
          'font.serif': ['Computer Modern Roman'],
          'ps.usedistiller': 'xpdf',
          'text.usetex': True,
          # include here any neede package for latex
          'text.latex.preamble': [r'\usepackage{amsmath}'],
          }
plt.rcParams.update(params)


raw_data = scipy.io.loadmat('wbcd.mat')

##Training data (e.g, wbcd dataset)
# ===========================================


#  Wisconsin breast cancer data
#  source: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
data = raw_data['wbcd']

#  binary label vector
labels =data[:,1].reshape(569,1)

spliting_persentage=30

nData = len(data) #number of datapoints available in the training data

train_data = data[:math.ceil((1-spliting_persentage/100)*nData)]
test_data = data[math.ceil((1-spliting_persentage/100)*nData):]

x=train_data[:,2:]
nData = len(train_data) #number of datapoints available in the training data
nx = len(x[0]) #dimension of the feature vector
labels =train_data[:,1].reshape(nData,1)

nData_test=len(test_data)

#  binary label vector
labels_test =test_data[:,1].reshape(nData_test,1)


#  features
x_test=test_data[:,2:]

nx_test = len(x_test[0]) #dimension of the feature vector


##Simulation parameters
# =====================================
# dimension of the parameter vector theta = (a, b, w)'

n_p=nx+2


nSample=1000 # number of samples


theta = np.genfromtxt('theta.dat')
rho = np.genfromtxt('rho.dat').reshape(nSample,1)


Risk_estimate1_beta_3 = np.genfromtxt('Risk_esstimate1_beta=0.03.dat')

Risk_estimate1_beta_5= np.genfromtxt('Risk_esstimate1_beta=0.05.dat')

Risk_estimate1_beta_7 = np.genfromtxt('Risk_esstimate1_beta=0.07.dat')

Risk_estimate2_beta_3=np.genfromtxt('Risk_estimate2_beta=0.03.dat')

Risk_estimate2_beta_5=np.genfromtxt('Risk_estimate2_beta=0.05.dat')

Risk_estimate2_beta_7=np.genfromtxt('Risk_estimate2_beta=0.07.dat')



#####################################################################################################################


Phi=(np.tanh(np.matmul(theta[:,2:],np.transpose(x_test))+theta[:,1].reshape(nSample,1)))*(theta[:,0].reshape(nSample,1))
f_hat=Phi.transpose().sum(axis=1).reshape(nData_test,1)/(nSample)
f_hat_estimate1=(np.exp(f_hat))/(1+np.exp(f_hat))

f_hat_weight=np.matmul(Phi.transpose(),rho)

f_hat_estimate2=np.zeros((nData_test)).reshape(nData_test,1)


for i in range(nData_test):
    if f_hat_weight[i]>0:
        f_hat_estimate2[i]=1
    if f_hat_weight[i]<0:
        f_hat_estimate2[i]=0



tol=1e-6

for i in range(nData_test):
    if f_hat_estimate1[i]>1-tol and f_hat_estimate1[i]<1+tol:
        f_hat_estimate1[i]=1
    if f_hat_estimate1[i]<tol and f_hat_estimate1[i]>-tol:
        f_hat_estimate1[i]=0


print("Accuracy of the Classification with beta=0.05 for estimate 1: ",(1/nData_test)*np.sum(f_hat_estimate1 == labels_test)*100)

print("Accuracy of the Classification with beta=0.05 for estimate 2: ",(1/nData_test)*np.sum(f_hat_estimate2 == labels_test)*100)


fig1 = plt.figure()
ax = fig1.add_subplot(2, 1, 1)
line, = ax.step(range(len(labels_test)),labels_test, color='black', lw=2,label="actual labels")
line, = ax.step(range(len(f_hat_estimate1)),f_hat_estimate1, color='red',linestyle='--', lw=1.5, label="estimate \#1")
ax.set_yticks([0,1])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.legend(markerscale=1.5, numpoints=1, loc='upper center', ncol=3,  prop={'size':15}, bbox_to_anchor=[0.75, .85], columnspacing=1.0, labelspacing=0.0, handletextpad=0.0, handlelength=2,fancybox=True, shadow=True)

ax = fig1.add_subplot(2, 1, 2)
line, = ax.step(range(len(labels_test)),labels_test, color='black', lw=2,label=" actual labels")
line, = ax.step(0,0, color='red',linestyle='--', lw=1.5, label="estimate \#1")
line, = ax.step(range(len(f_hat_estimate2)),f_hat_estimate2, color='blue',linestyle='-.', lw=1.5,  label="estimate \#2")
ax.set_yticks([0,1])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend(markerscale=1.5, numpoints=1, loc='upper center', ncol=3,frameon=False,  prop={'size':15}, bbox_to_anchor=[0.5, 2.4], columnspacing=1.2, labelspacing=0.1, handletextpad=0.5, handlelength=2)
fig1.set_size_inches(7, 5 )
ax.set_xlabel(" The test data index",fontsize=15)
plt.savefig('f_hat_WBDC.png', dpi=300)


###################################################################################################################################################

a1 = np.minimum.reduce((Risk_estimate1_beta_3, Risk_estimate1_beta_5,Risk_estimate1_beta_7))
a2 = np.maximum.reduce((Risk_estimate1_beta_3, Risk_estimate1_beta_5,Risk_estimate1_beta_7))



a3 = np.minimum.reduce((Risk_estimate2_beta_3,Risk_estimate2_beta_5,Risk_estimate2_beta_7))
a4 = np.maximum.reduce((Risk_estimate2_beta_3,Risk_estimate2_beta_5,Risk_estimate2_beta_7))






x = np.arange(len(Risk_estimate1_beta_3))


fig2 = plt.figure()
ax = fig2.add_subplot(2, 1, 1)

line, = ax.loglog(Risk_estimate1_beta_5, color='red', lw=1.5,linestyle='--',label="estimate \#1")
line, = ax.step(0,0, color='blue', lw=1.5, label="estimate \#2")

plt.fill_between(x, a1, a2, color='red',alpha=.3)


ax.set_ylabel(r"Risk functional $R_{\beta}$",fontsize=15)
ax.yaxis.set_label_coords(-0.11,-.1)



ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend(markerscale=1.5, numpoints=1, loc='upper center', ncol=2,  prop={'size':15}, frameon=False,bbox_to_anchor=[0.5, 1.2], columnspacing=1.2, labelspacing=0.1, handletextpad=0.5, handlelength=2)
ax.grid(True, which="both", ls="-", color='0.8')
ax.set_yticks([.01,.02,.03])
ax = fig2.add_subplot(2, 1, 2)

line, = ax.loglog(Risk_estimate2_beta_5, color='blue', lw=1.5, label="estimate \#2")

plt.fill_between(x, a3, a4, color='blue',alpha=.3)


# ax.set_ylabel(r"Risk functional $R$",fontsize=15)
ax.yaxis.set_label_coords(-0.11,0.45)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.legend(markerscale=1.5, numpoints=1, loc='upper center', ncol=2,  prop={'size':15}, frameon=False,bbox_to_anchor=[0.5, 1.1], columnspacing=1.2, labelspacing=0.0, handletextpad=0.2, handlelength=2)
ax.grid(True, which="both", ls="-", color='0.8')
fig2.set_size_inches(8, 5 )
ax.set_xlabel(" Iterations",fontsize=15)
plt.savefig('Risk_WBDC.png', dpi=300)




