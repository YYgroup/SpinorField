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

    def forward(self,phi):
        #A=torch.tensor([[cos(self.para[0]/2),exp(-1j*self.para[1])*sin(self.para[0]/2)],[exp(1j*self.para[2])*sin(self.para[0]/2),exp(1j*(self.para[1]+self.para[2]))*cos(self.para[0]/2)]],dtype=torch.complex128)
        #A=torch.zeros(2,2,dtype=torch.complex128)
        #A[0,0]=cos(self.para[0]/2)
        #A[0,1]=-exp(1j*self.para[1])*sin(self.para[0]/2)
        #A[1,0]=exp(1j*self.para[2])*sin(self.para[0]/2)
        #A[1,1]=exp(1j*(self.para[1]+self.para[2]))*cos(self.para[0]/2)
        A1=torch.cat([cos(self.para0/2),-exp(1j*self.para1)*sin(self.para0/2)],dim=1)
        A2=torch.cat([exp(1j*self.para2)*sin(self.para0/2),exp(1j*(self.para1+self.para2))*cos(self.para0/2)],dim=1)
        A=torch.cat([A1,A2],dim=0)
        phi_t=phi.transpose(self.N_total-self.N_ctrl-1,0).transpose(self.N_total-self.N_ctrl-1,N_qubit-2)
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
        #phi_temp=A@phi_t[:1]
        return torch.cat((A@phi_t[:1],phi_t[1:]),0).transpose(self.N_total-self.N_ctrl-1,N_qubit-2).transpose(self.N_total-self.N_ctrl-1,0)

class com_U3_tensor(nn.Module):
    def __init__(self,N_total):
        super().__init__()

        #self.S=nn.ModuleList([U3_tensor(N_total,N_total-1,i) for i in range(0,N_total-1)])
        self.S=nn.ModuleList([])
        for i in range(0,N_total-1):
            self.S.append(U3_tensor(N_total,N_total-1,i))
            self.S.append(U3_tensor_nega(N_total,N_total-1,i))

    def forward(self,phi):
        for a,b in enumerate(self.S):
            phi=b(phi)
        return phi

    

N_qubit=6
N_point=1<<(N_qubit-1)
pi=np.pi
L=2*pi
dx=L/N_point
X=torch.linspace(dx/2,L-dx/2,N_point)
u=sin(X)
eta=u*dx
eta=eta.to(device)
unified=np.sqrt(1<<(N_qubit-1))

hbar=1
epsilon=1


def loss_fn_Chern(phi):
    phi=phi*unified
    phi=phi.reshape(1<<N_qubit)
    delta1=exp(-1j*eta/2/hbar)*(phi[0:N_point].roll(-1))-exp(1j*eta/2/hbar)*(phi[0:N_point])
    delta2=exp(-1j*eta/2/hbar)*(phi[N_point:2*N_point].roll(-1))-exp(1j*eta/2/hbar)*(phi[N_point:2*N_point])
    AB=phi[0:N_point]
    CD=phi[N_point:2*N_point]

    s1=torch.conj(AB)*AB-torch.conj(CD)*CD
    s2=2*AB*torch.conj(CD)
    s1=(s1+s1.roll(-1))/2
    s2=(s2+s2.roll(-1))/2
    delta3=(1+epsilon+(1-epsilon)*s1)*delta1+(1-epsilon)*s2*delta2
    delta4=(1-epsilon)*torch.conj(s2)*delta1+(1+epsilon-(1-epsilon)*s1)*delta2
    cost=torch.norm(delta3,2)**2+torch.norm(delta4,2)**2
    return cost

def phi_to_U(phi):
    phi=phi.reshape(1<<N_qubit)
    AB=phi[0:N_point]
    CD=phi[N_point:N_point*2]
    A=np.real(AB)
    B=np.imag(AB)
    C=np.real(CD)
    D=np.imag(CD)

    DA=(torch.roll(A,-1)-A)/dx
    DB=(torch.roll(B,-1)-B)/dx
    DC=(torch.roll(C,-1)-C)/dx
    DD=(torch.roll(D,-1)-D)/dx
    

    U=hbar*(A*DB-B*DA+C*DD-D*DC)
    return U

def loss_fn(phi):
    U=phi_to_U(phi*unified)
    return torch.norm(U-u,2)**2


model=torch.nn.Sequential(
    com_U3_tensor(N_qubit),
    com_U3_tensor(N_qubit)
)
model.to(device)

optimizer=torch.optim.AdamW(model.parameters(),lr=0.05)
step=1000
output_step=100
real=torch.zeros(1<<N_qubit,dtype=torch.float64)
imag=torch.zeros(1<<N_qubit,dtype=torch.float64)
temp=np.sqrt(1/(1<<(N_qubit-1)))
for i in range(0,1<<(N_qubit-1)):
    real[i]=temp
input=torch.complex(real,imag)
input=input.reshape(2,2,2,2,2,2)
input.requires_grad_()
input=input.to(device)
LOSS=torch.zeros(step)
LOSS.requires_grad=False

flag=True

for t in range(1,step+1):
    sample=model(input)
    loss=loss_fn_Chern(sample)
    LOSS[t-1]=loss.item()
    if(flag and loss<0.1):
        for param_group in optimizer.param_groups:
            param_group['lr']=0.015
            flag=False
    if(t%50==0):
        print('No.{: 5d}, loss {: 6f}'.format(t,loss.item()))
    if (t%100==0):
        epsilon=epsilon*0.2
    if(t%output_step==0):
        filename='torch1d_'+str(t)+'.mat'
        phi=sample.detach()*unified
        U=phi_to_U(phi)
        savemat(filename,{'phi':np.array(phi.cpu().reshape(1,1<<N_qubit)),'U':np.array(U.cpu()),'u0':np.array(u.cpu()),'loss':LOSS.detach()})
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()











