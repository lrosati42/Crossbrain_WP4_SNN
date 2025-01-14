'''
Created on 20 juil. 2012

@author: Squirel
'''

from epileptor import fast_synapse_T as fast_synapse 
from epileptor import slow_synapse_T as slow_synapse
from random import uniform
import numpy as np

class pop2n(object):
    '''
    classdocs
    '''


    def __init__(self, aa=8., tau=20., z0=0., I2=0.9, CpES=1., CpCS=0.9, g_x2x2=0.5, g_x1x2=0.8, g_x2x2_slow=0., c2=0.3, noise=0.1):
        '''
        Constructor
        '''
        self.parameters = aa, tau,I2,z0
        self.x2=0. ; self.y2=0.; self.g=0.;
        self.c2=0.3
        self.noise=noise
        self.y2_plot=[]
        
        #synapses management syntax  syn_ pre post
        self.syn_x2x2 = fast_synapse.fast_synapse(g_x2x2, -80, 5, 0.18, 2)  #gbar, E, alpha, beta, Vt
        self.syn_x1x2 = fast_synapse.fast_synapse(g_x1x2, 0, 1.1, 0.19, 2)  #gbar, E, alpha, beta, Vt
        self.syn_x2x2_slow = slow_synapse.slow_synapse(g_x2x2_slow, -95, 0.09, 0.0012, 2)  #gbar, E, alpha, beta, Vt
        self.CpES = CpES # electrical synapses coupling
        self.CpCS = CpCS # chemical synapses coupling
        
        #plotting management
        self.Isyn_x1x2_plot = []
        self.Isyn_x2x2_plot = []
        self.Isyn_x2x2_slow_plot = []
        self.Igj_x2x2_plot = []
        
        # input coupling management
        self.pop1in=[]
        self.pop2in=[]
        self.pop2in_gj=[]
        
        
        #for the morris lecar implementation
        self.V1=-1.2; self.V2=18; 
        #self.V3=2; self.V4=30; self.phi=0.04;     #Hopf parameters
        self.V3=12; self.V4=17.4; self.phi=0.067; #SNLC parameters
        #self.V3=12; self.V4=17.4; self.phi=0.23; #Homoclinic parameters
        self.gCa_bar=4; # !!! must be 4.0 in SNLC & Homoclinic settings 
        
        self.gK_bar=8.0; self.gL_bar=2; 
        self.EK=-84; self.EL=-60; self.ECa=120;   
        self.Cm=20;
        self.n_plot=[]
        
       
    def euler(self, dt, Iin, x1bar, x2bar, zbar, ti):
        aa,tau,I2,z0 = self.parameters
        
        #get average inputs
        x2in=0 #from pop2 synapses
        for n in self.pop2in:
            x2in+=n.x2
        x2in=x2in/len(self.pop2in)
        
        x1in=0 #from pop1 synapses
        for n in self.pop1in:
            x1in+=n.x1
        x1in=x1in/len(self.pop1in)
        
        x2in_gj=0 #from pop2 gap junctions
        for n in self.pop2in_gj:
            x2in_gj+=n.x2
        x2in_gj=x2in_gj/len(self.pop2in_gj)
        
        Isyn_x2x2 = self.syn_x2x2.eq(x2in*50, self.x2*50, dt, ti*dt)
        #self.Isyn_x2x2_plot.append(Isyn_x2x2/50)
        
        Isyn_x2x2_slow = self.syn_x2x2_slow.eq(x2in*50, self.x2*50, dt, ti*dt)
        #self.Isyn_x2x2_slow_plot.append(Isyn_x2x2_slow/50)
        
        Isyn_x1x2 = self.syn_x1x2.eq(x1in*50, self.x2*50, dt, ti*dt)
        #self.Isyn_x1x2_plot.append(Isyn_x1x2/50)
        
        #self.Igj_x2x2_plot.append(self.CpES*(x2in_gj-self.x2))
        
        """x2next = self.x2 + (-self.y2 + self.x2 - self.x2**3 + I2 - (zbar-3.5)*0.3 + self.CpCS*Isyn_x2x2/50 + self.CpES*(x2in_gj-self.x2) + Isyn_x1x2/50)*dt#+ 2*self.g - (zbar-3.5)*0.3 + self.CpES*(x2bar-self.x2) + self.CpCS*self.syn_x2x2.eq(x2bar, self.x2, dt, ti*dt))*dt 
        if self.x2 < -0.25:
            y2next = self.y2 + (- 1./tau * self.y2)*dt
        else:
            y2next = self.y2 + (1./tau * (-self.y2 + aa*(self.x2 + 0.25)))*dt
        
        self.x2, self.y2 = x2next, y2next
        self.y2_plot.append(y2next)"""
        
        V=self.x2*20; n=self.y2;
        I_in=I2*50 - self.c2*(zbar-3)*50 + self.CpCS*Isyn_x2x2 + self.CpES*(x2in_gj-self.x2)*50 + Isyn_x1x2 + Isyn_x2x2_slow # - (zbar-3)*0.3*50
        
        m_inf=0.5*(1+np.tanh((V-self.V1)/self.V2))
        n_inf=0.5*(1+np.tanh((V-self.V3)/self.V4))
        tau_n=1/np.cosh((V-self.V3)/(2*self.V4))
                    
        n_next = n + (self.phi*(n_inf-n)/tau_n)*dt; #slow repolarizing K+
        V_next = V + ((I_in - self.gCa_bar*m_inf*(V-self.ECa) - self.gK_bar*n*(V-self.EK) - self.gL_bar*(V-self.EL))/self.Cm + uniform(-self.noise,self.noise)*50)*dt; #voltage
        
        self.n_plot.append(n_next) 
        
        self.x2=V_next/20; self.y2=n_next
        return V_next/20,n_next  
    
    def get_nullclines(self,x2span):
        aa,tau,z0,I2 = self.parameters
        Vspan=x2span
        V_nullcline=[]
        n_nullcline=[]
        for V in Vspan:
            m_inf=0.5*(1+np.tanh((V-self.V1)/self.V2))
            n_inf=0.5*(1+np.tanh((V-self.V3)/self.V4))
            V_nullcline.append((I2*50 - self.gCa_bar*m_inf*(V-self.ECa) - self.gL_bar*(V-self.EL))/(self.gK_bar*(V-self.EK)))
            n_nullcline.append(n_inf)
        return np.array(V_nullcline), np.array(n_nullcline)
    
    def connect_syn_pop2n(self, neuron_list):
        for neuron in neuron_list:
            self.pop2in.append(neuron)
            
    def connect_syn_pop1n(self, neuron_list):
        for neuron in neuron_list:
            self.pop1in.append(neuron)
            
    def connect_gap(self, neuron_list):
        for neuron in neuron_list:
            self.pop2in_gj.append(neuron)