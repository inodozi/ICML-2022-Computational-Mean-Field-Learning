def Drift():

    x_torch = torch.tensor(x)
    y_torch = torch.tensor(y)


    rho= torch.tensor(rho_prev)


    w= torch.tensor(theta_prev[:,2:],requires_grad = True)
    a= torch.tensor(theta_prev[:,0].reshape(nSample,1),requires_grad = True)
    b= torch.tensor(theta_prev[:,1].reshape(nSample,1),requires_grad = True)

    Phi_torch = a*(torch.tanh(w.mm(torch.transpose(x_torch,0,1))+b))
    U=(1/nData)*Phi_torch.mm(torch.transpose(Phi_torch, 0, 1))
    W=U.mm(rho)

    V=(-2/nData)*Phi_torch.mm(y_torch)

    drift=V+W
    one=torch.ones((nSample,1))
    drift.backward(torch.FloatTensor(one))

    Phi=(np.tanh(np.matmul(theta_prev[:,2:],np.transpose(x))+theta_prev[:,1].reshape(nSample,1)))*(theta_prev[:,0].reshape(nSample,1))
    driftk = np.hstack((-a.grad.numpy(),-b.grad.numpy(),-w.grad.numpy()))
    return driftk, Phi

#####################################################################################################################
def FixedPointIteration():

    tol=1e-3  #tolerance
    maxiter=300 # max number of iterations for k fixed

    #     squared distance matrix
    C=distance.cdist(theta_current,theta_prev,'euclidean')**2
    #      Find previous value of psi function
    V_prev=(-2/nData)*np.matmul(Phi,y)

    #       Find previous value of phi function
    U_prev=(1/nData) * np.matmul(Phi , Phi.transpose())

    #     elementwise exponential of a matrix
    gamma = np.exp(-C/(2*epsilon));
    #     elementwise exponential of a vector
    xi=np.exp(-beta*(V_prev)-beta*np.matmul(U_prev,rho_prev)-np.ones((nSample,1)))

    lambda_1=np.random.rand(nSample,1)
    z0=np.exp(lambda_1*h/epsilon)
    z =np.hstack((z0,np.zeros((nSample,maxiter - 1))))

    yy=np.hstack((rho_prev/(np.matmul(gamma,z0)),np.zeros((nSample,maxiter - 1))))

    l=0
    while l < maxiter-1:
        z[:,l+1]=np.power(xi/(np.matmul(gamma.transpose(),yy[:,l].reshape(nSample,1))),1/(1+(beta*epsilon/h)))[:,0]
        yy[:,l+1]=(rho_prev/(np.matmul(gamma,z[:, l+1].reshape(nSample,1))))[:,0]
        if ((LA.norm(yy[:, l+1]-yy[:, l]) < tol) and (LA.norm(z[:, l+1]-z[:, l]) < tol)):
            break
        l+=1
    rho_next=z[:,l].reshape(nSample,1)*(np.matmul(gamma.transpose(),yy[:,l].reshape(nSample,1)))
    return rho_next, V_prev, U_prev

#####################################################################################################################
def EulerMaruyama_MeanField():
    gdw =np.sqrt(h*2/beta)*np.random.randn(nSample,n_p)
    theta_current=theta_prev+(h*driftk)+gdw
    return theta_current

#####################################################################################################################

import numpy as np
import torch
from numpy import linalg as LA
from ttictoc import tic,toc
from scipy.spatial import distance
import random
import math
import more_itertools as mit
import matplotlib.pyplot as plt
import pylab

import scipy.io
raw_data = scipy.io.loadmat('wbcd.mat')

##Training data (e.g, wbcd dataset)
# ===========================================


#  Wisconsin breast cancer data
#  source: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
data = raw_data['wbcd']
nData = len(data) #number of datapoints available in the training data

# random.shuffle(data)

spliting_persentage=30

train_data = data[:math.ceil((1-spliting_persentage/100)*nData)]
test_data = data[math.ceil((1-spliting_persentage/100)*nData):]
nData_test=len(test_data)

data=train_data
nData=len(train_data)


#  binary label vector
labels =data[:,1].reshape(nData,1)


# rescale labels from {0,1} to {-1,1}
ymin, ymax= -1, 1
y = ymin*np.ones((len(data),1)) + (ymax - ymin)*labels
#  features
x=data[:,2:]

nx = len(x[0]) #dimension of the feature vector


#  binary label vector
labels_test =test_data[:,1].reshape(nData_test,1)


# rescale labels from {0,1} to {-1,1}
y_test = ymin*np.ones((len(test_data),1)) + (ymax - ymin)*labels_test
#  features
x_test=test_data[:,2:]

nx_test = len(x_test[0]) #dimension of the feature vector



##Simulation parameters
# =====================================
# dimension of the parameter vector theta = (a, b, w)'

n_p=nx+2

# parameters for the proximal recursion

nSample=1000 # number of samples
beta=.05 # inverse temperature
epsilon=1  #entropic regularizing coefficient
h=1e-3 # time step
numSteps=int(1e6) #1e6 # number of steps k, in discretization t=kh
## propagate joint PDF over the parameter space
#  ===============================================
    # parameter ranges
a_min, a_max=0.9, 1.1; # min-max scaling
b_min, b_max=-0.1, 0.1; # min-max bias
w_min, w_max=-1, 1;
numRandomRun=1 # 50



Risk_estimate1=np.zeros((numSteps)).reshape(numSteps,1)
Risk_estimate2=np.zeros((numSteps)).reshape(numSteps,1)

comptime=np.zeros((numRandomRun,1))
F0=np.sum(y**2)/nData


for r in range(numRandomRun):


#      generate uniformly random initial parameters
    a0=a_min+(a_max-a_min)*np.random.rand(nSample,1)
    b0=b_min+(b_max-b_min)*np.random.rand(nSample,1)
    w0=w_min+(w_max-w_min)*np.random.rand(nSample,nx)

    #      concatenate
    theta0=np.hstack((a0,b0,w0))

    #     evaluate initial uniform joint PDF at those samples


    rho0=np.ones((nSample,1))*(1/((a_max-a_min)*(b_max-b_min)*((w_max-w_min)**nx)))
    #     preallocate
    theta_prev=theta0
    rho_prev = rho0
    
    
    tol_Risk_estimate1=1e-3 # numbers coloser than this number to zero and one 
                  # in the Risk_estimate1 and f_hat_estimate1 will be considered as zero and one, respectively.

    f_hat_estimate2=np.zeros((nData)).reshape(nData,1)
    f_hat_estimate1=np.random.rand((nData)).reshape(nData,1)

    tic()
    #
    for k in range(numSteps):
        if (k%5e3 == 0):
            print("Now, running iterations between k=",k+1,"and k=", k+int(5e3))
            print("And, the Risk_estimate1 value at iteration #",k, "was:",Risk_estimate1[k-1])
        driftk, Phi=Drift()
        #        update of the parameter samples
        theta_current=EulerMaruyama_MeanField()
        #       proximal update of the joint PDF
        rho_next, V_prev, U_prev=FixedPointIteration()
        theta_prev=theta_current
        rho_prev=rho_next

        f_hat=Phi.transpose().sum(axis=1).reshape(nData,1)/nSample
        f_hat_estimate1=(np.exp(f_hat))/(1+np.exp(f_hat))
        ## computing the Risk_estimate1 function with weighted f_hat for test data
        f_hat_weight=np.matmul(Phi.transpose(),rho_prev)  
        for i in range(nData):#pulishing f_hat_estimate1 and Risk_estimate1 around zero and 1 and make them exactly zero and one
            if f_hat_estimate1[i]>1-tol_Risk_estimate1:
                f_hat_estimate1[i]=1
            if f_hat_estimate1[i]<tol_Risk_estimate1:
                f_hat_estimate1[i]=0
            if f_hat_weight[i]>0:
                f_hat_estimate2[i]=1
            if f_hat_weight[i]<0:
                f_hat_estimate2[i]=0
        Risk_estimate1[k]=(0.5/nData)*LA.norm(labels-f_hat_estimate1)
        Risk_estimate2[k]=(0.5/nData)*LA.norm(labels-f_hat_estimate2)
    comptime[r]=toc()
print("Computation time: "+str(comptime))

np.savetxt('Risk_estimate1 '+'beta='+str(beta)+'.dat', Risk_estimate1)
np.savetxt('Risk_estimate2 '+'beta='+str(beta)+'.dat', Risk_estimate2)

np.savetxt('theta'+str(numSteps)+'.dat', theta_prev)
np.savetxt('rho'+str(numSteps)+'.dat', rho_prev)


