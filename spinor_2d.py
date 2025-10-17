import numpy as np
import torch
from scipy.io import savemat, loadmat
import time

nn=torch.nn
sin=torch.sin
cos=torch.cos
exp=torch.exp

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device=torch.device("cpu")
print(device)


class U3_tensor(nn.Module):
    def __init__(self,N_total,N_tar,N_ctrl):
        super().__init__()

        self.para0=torch.nn.Parameter(torch.randn(1,1,dtype=torch.float64))
        self.para0.requires_grad=True
        self.para1=torch.nn.Parameter(torch.randn(1,1,dtype=torch.float64))
        self.para1.requires_grad=True
        self.para2=torch.nn.Parameter(torch.randn(1,1,dtype=torch.float64))
        self.para2.requires_grad=True
        self.N_total=N_total
        self.N_tar=N_tar
        self.N_ctrl=N_ctrl

    #@profile(precision=4,stream=open('memory_profiler.log','w+'))  
    def forward(self,phi):#this forward method only works for N_tar=N_total-1
        A1=torch.cat([cos(self.para0/2),-exp(1j*self.para1)*sin(self.para0/2)],dim=1)
        A2=torch.cat([exp(1j*self.para2)*sin(self.para0/2),exp(1j*(self.para1+self.para2))*cos(self.para0/2)],dim=1)
        A=torch.cat([A1,A2],dim=0)#the unitary matrix decided by self.para0 to self.para3
        phi_t=phi.transpose(self.N_total-self.N_ctrl-1,0).transpose(self.N_total-self.N_ctrl-1,N_qubit-2)#move the numbers that need to transformed by A in phi_t[1:], and in certain order
        return torch.cat((phi_t[:1],A@phi_t[1:]),0).transpose(self.N_total-self.N_ctrl-1,N_qubit-2).transpose(self.N_total-self.N_ctrl-1,0)


class U3_tensor_nega(nn.Module):
    def __init__(self,N_total,N_tar,N_ctrl):
        super().__init__()

        self.para0=torch.nn.Parameter(torch.randn(1,1,dtype=torch.float64))
        self.para0.requires_grad=True
        self.para1=torch.nn.Parameter(torch.randn(1,1,dtype=torch.float64))
        self.para1.requires_grad=True
        self.para2=torch.nn.Parameter(torch.randn(1,1,dtype=torch.float64))
        self.para2.requires_grad=True
        self.N_total=N_total
        self.N_tar=N_tar
        self.N_ctrl=N_ctrl

    def forward(self,phi):
        #A=torch.tensor([[cos(self.para[0]/2),-exp(1j*self.para[1])*sin(self.para[0]/2)],[exp(1j*self.para[2])*sin(self.para[0]/2),exp(1j*(self.para[1]+self.para[2]))*cos(self.para[0]/2)]],dtype=torch.complex128)
        A1=torch.cat([cos(self.para0/2),-exp(1j*self.para1)*sin(self.para0/2)],dim=1)
        A2=torch.cat([exp(1j*self.para2)*sin(self.para0/2),exp(1j*(self.para1+self.para2))*cos(self.para0/2)],dim=1)
        A=torch.cat([A1,A2],dim=0)
        phi_t=phi.transpose(self.N_total-self.N_ctrl-1,0).transpose(self.N_total-self.N_ctrl-1,N_qubit-2)
        return torch.cat((A@phi_t[:1],phi_t[1:]),0).transpose(self.N_total-self.N_ctrl-1,N_qubit-2).transpose(self.N_total-self.N_ctrl-1,0)

class com_U3_tensor(nn.Module):#a class that enumerate N_ctrl from 0 to N_total-2
    def __init__(self,N_total):
        super().__init__()

        self.S=nn.ModuleList([])
        for i in range(0,N_total-1):
            self.S.append(U3_tensor(N_total,N_total-1,i))
            self.S.append(U3_tensor_nega(N_total,N_total-1,i))

    def forward(self,phi):
        for a,b in enumerate(self.S):
            phi=b(phi)
        return phi

    
N_dim=5#for 2d problem, each dimension contains 2^N_dim grid points
N_qubit=2*N_dim+1#number of qubits needed
N_point=1<<(N_dim)#grid points of one dimension
N_total=1<<(N_qubit-1)#total number of grid points
pi=np.pi
L=2*pi
dx=L/N_point
X1=torch.linspace(dx/2,L-dx/2,N_point,dtype=torch.float64)
X2=torch.linspace(0,L-dx,N_point,dtype=torch.float64)
XV1,YV1=torch.meshgrid(X1,X2)
XV2,YV2=torch.meshgrid(X2,X1)
UX=sin(XV1)*cos(YV1)
UY=cos(XV2)*sin(YV2)
etax=UX*dx
etay=UY*dx#1-form of the same target velocity, \eta=u\mathrm{d}x
etax=etax.to(device)
etay=etay.to(device)
unified=np.sqrt(1<<(N_qubit-1))#normalization coefficient

hbar=1
epsilon=1

def loss_fn(phi):
    phi=phi*unified
    phi=phi.reshape(1<<N_qubit)
    [U1,U2]=phi_to_U(phi)

    cost=torch.norm(U1-UX,2)+torch.norm(U2-UY,2)
    return cost


def loss_fn_Chern(phi):#loss function proposed by A.Chern
    phi=phi*unified
    phi=phi.reshape(1<<N_qubit)
    phi1=phi[0:N_total].view(N_point,N_point)
    phi2=phi[N_total:2*N_total].view(N_point,N_point)
    delta1x=exp(-1j*etax/2/hbar)*(phi1.roll(-1,dims=0))-exp(1j*etax/2/hbar)*(phi1)
    delta1y=exp(-1j*etay/2/hbar)*(phi1.roll(-1,dims=1))-exp(1j*etay/2/hbar)*(phi1)
    delta2x=exp(-1j*etax/2/hbar)*(phi2.roll(-1,dims=0))-exp(1j*etax/2/hbar)*(phi2)
    delta2y=exp(-1j*etay/2/hbar)*(phi2.roll(-1,dims=1))-exp(1j*etay/2/hbar)*(phi2)

    s1=torch.conj(phi1)*phi1-torch.conj(phi2)*phi2
    s2=2*phi1*torch.conj(phi2)
    s1x=(s1+s1.roll(-1,dims=0))/2
    s1y=(s1+s1.roll(-1,dims=1))/2
    s2x=(s2+s2.roll(-1,dims=0))/2
    s2y=(s2+s2.roll(-1,dims=1))/2
    delta3x=(1+epsilon+(1-epsilon)*s1x)*delta1x+(1-epsilon)*s2x*delta2x
    delta3y=(1+epsilon+(1-epsilon)*s1y)*delta1y+(1-epsilon)*s2y*delta2y
    delta4x=(1-epsilon)*torch.conj(s2x)*delta1x+(1+epsilon-(1-epsilon)*s1x)*delta2x
    delta4y=(1-epsilon)*torch.conj(s2y)*delta1y+(1+epsilon-(1-epsilon)*s1y)*delta2y

    cost=torch.norm(delta3x,2)**2+torch.norm(delta3y,2)**2+torch.norm(delta4x,2)**2+torch.norm(delta4y,2)**2
    return cost

def phi_to_U(phi):#calculate velocity field from phi in 2d case
    phi=phi.reshape(1<<N_qubit)
    AB=phi[0:N_total].view(N_point,N_point)
    CD=phi[N_total:N_total*2].view(N_point,N_point)
    A=np.real(AB)
    B=np.imag(AB)
    C=np.real(CD)
    D=np.imag(CD)

    DAx=(torch.roll(A,-1,dims=0)-A)/dx
    DBx=(torch.roll(B,-1,dims=0)-B)/dx
    DCx=(torch.roll(C,-1,dims=0)-C)/dx
    DDx=(torch.roll(D,-1,dims=0)-D)/dx
    DAy=(torch.roll(A,-1,dims=1)-A)/dx
    DBy=(torch.roll(B,-1,dims=1)-B)/dx
    DCy=(torch.roll(C,-1,dims=1)-C)/dx
    DDy=(torch.roll(D,-1,dims=1)-D)/dx

    Amx=(torch.roll(A,-1,dims=0)+A)/2
    Bmx=(torch.roll(B,-1,dims=0)+B)/2
    Cmx=(torch.roll(C,-1,dims=0)+C)/2
    Dmx=(torch.roll(D,-1,dims=0)+D)/2
    Amy=(torch.roll(A,-1,dims=1)+A)/2
    Bmy=(torch.roll(B,-1,dims=1)+B)/2
    Cmy=(torch.roll(C,-1,dims=1)+C)/2
    Dmy=(torch.roll(D,-1,dims=1)+D)/2
    

    Ux=hbar*(Amx*DBx-Bmx*DAx+Cmx*DDx-Dmx*DCx)
    Uy=hbar*(Amy*DBy-Bmy*DAy+Cmy*DDy-Dmy*DCy)
    return Ux,Uy 


model=torch.nn.Sequential(
    com_U3_tensor(N_qubit),
    com_U3_tensor(N_qubit),
    com_U3_tensor(N_qubit),
    com_U3_tensor(N_qubit),
    com_U3_tensor(N_qubit)
)
model.to(device)

optimizer=torch.optim.AdamW(model.parameters(),lr=5e-3)
#scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',threshold=5e-5,verbose=True,cooldown=200,factor=0.5)
step=10000
output_step=1000#generate a result file for every output_step steps
jump_step=1000
real=torch.zeros(1<<N_qubit,dtype=torch.float64)
imag=torch.zeros(1<<N_qubit,dtype=torch.float64)
temp=np.sqrt(1/(1<<(N_qubit-1)))
for i in range(0,1<<(N_qubit-1)):
    real[i]=temp
input=torch.complex(real,imag)#an initial input so that every complex number is \sqrt{2}/2+0j when normalized
input=input.reshape(2,2,2,2,2,2,2,2,2,2,2)
input.requires_grad_()
input=input.to(device)
LOSS=torch.zeros(step)
LOSS.requires_grad=False#record the loss value

flag=True#for changing the study rate, a simple version

for t in range(1,step+1):
    sample=model(input)
    loss=loss_fn_Chern(sample)
    LOSS[t-1]=loss.item()
    if(t%100==0):
        print('No.{: 5d}, loss {: 6f}'.format(t,loss.item()))
    if (t%jump_step==0):#epsilon needs to decrease towards 0, as a part of algorithm of A. Chern
        epsilon=epsilon*0.1
        if(loss<10):
            for param_group in optimizer.param_groups:
                param_group['lr']=3e-4
    if(t%output_step==0):
        filename='torch2d_'+str(t)+'.mat'
        phi=(sample.detach()*unified).reshape(1<<N_qubit)
        Ux,Uy=phi_to_U(phi)
        savemat(filename,{'AB':np.array(phi[0:N_point*N_point].cpu().reshape(N_point,N_point)),'CD':np.array(phi[N_point*N_point:2*N_point*N_point].cpu().reshape(N_point,N_point)),'Ux':np.array(Ux.cpu()),'Uy':np.array(Uy.cpu()),'ux0':np.array(UX.cpu()),'uy0':np.array(UY.cpu()),'loss':LOSS.detach()})
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()









