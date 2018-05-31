# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from brian2 import *
start_scope()

simtime = 1*second # Simulation time

ed = -38*mV         #controls the position of the threshold
dd = 6*mV           #controls the sharpness of the threshold 

C_s = 370*pF ; C_d = 170*pF      #capacitance 
el = -70*mV                                              #reversal potential 
c_d = 2600*pA
aw_s = aw_a = aw_b = 0 ; aw_d = -13*nS                   #strength of subthreshold coupling 
bw_s = -200*pA ; bw_d = 0 ; bw_a = bw_b = -150*pA        #strength of spike-triggered adaptation 
tauw_s = 100*ms ; tauw_d = 30*ms       #time scale of the recovery variable 
tau_s = 16*ms ; tau_d = 7*ms; tau_a = 20*ms; tau_b = 10*ms # time scale of the membrane potential 

g_s = 1300*pA ; g_d = 1200*pA #models the regenrative activity in the dendrites 

#I_s = 460*pA ; I_d = 200*pA
I_s = 460*pA #; I_d = 300*pA

n = 100 ; p_1= [] ; p_2 = []; k = []

sigma=2*mV; tau=5*ms

 
for f in range(0,400):
    I_d = f*pA
    
###########################################
    
    eqs_1 ='''
    dv_s/dt = ( -(v_s-el)/tau_s ) + ( g_s * (1/(1+exp(-(v_d-ed)/dd))) + I_s + w)/C_s +  sigma * (2 / tau)**.5 *xi : volt (unless refractory)
    dw/dt = -w/tauw_s :amp
    dv_d/dt =  ( -(v_d-el)/tau_d ) + ( g_d*(1/(1+exp(-(v_d-ed)/dd))) +c_d*K + I_d + ws + g_i*(-(v_d-el)) )/C_d : volt 
    dws/dt = ( -ws + aw_d *(v_d - el))/tauw_d :amp
    K :1
    g_i : siemens
    '''   
    P_1 = PoissonGroup(1, 0*Hz)
    
    G1 = NeuronGroup(n, model=eqs_1,  threshold='v_s > -50*mV', reset='v_s =el ; w+= bw_s ',refractory=3*ms,method='euler') #+delay 
    
    G1.v_s = el
    G1.v_d = el
    
    backprop = Synapses(G1, G1, 
                         on_pre={'up': 'K += 1', 'down': 'K -=1'}, 
                         delay={'up': 0.5*ms, 'down': 2.5*ms}) 
    
    backprop.connect(j='i') # Connect all neurons to themselves 
    
    Y = Synapses(P_1, G1, on_pre='g_i+=1*nsiemens')
    
    Y.connect()
    
    #soma_1 = StateMonitor(G1,'v_s',record=[4])
    #dend_1 = StateMonitor(G1,'v_d',record=[4])
    #
    #kent = StateMonitor(G1,'K',record=True)
    #
    M_1 = SpikeMonitor(G1)
    
    ######################################################
    #
    P_2 = PoissonGroup(1, 0*Hz)
    
    G2 = NeuronGroup(n, model=eqs_1,  threshold='v_s > -50*mV', reset='v_s =el ; w+= bw_s ',refractory=3*ms,method='euler') #+delay 
    
    G2.v_s = el
    G2.v_d = el
    
    
    backprop = Synapses(G2, G2, 
                         on_pre={'up': 'K += 1', 'down': 'K-=1'}, 
                         delay={'up': 0.5*ms, 'down': 2.5*ms}) 
    
    backprop.connect(j='i') # Connect all neurons to themselves 
    
    Y_2 = Synapses(P_2, G2, on_pre='g_i+=1*nsiemens')
    
    Y_2.connect()
    
    #soma_2 = StateMonitor(G2,'v_s_2',record=[4])
    #dend_2 = StateMonitor(G2,'v_d_2',record=[4])
    #
    #kent = StateMonitor(G2,'K',record=True)
    #
    M_2 = SpikeMonitor(G2)


    run(simtime)
    
    p_1.append(M_1.count/simtime)
    p_2.append(M_2.count/simtime)
    k.append(f)


######################################################

 
#plot(M_1.t/ms, M_1.i, '.r')
#plot(M_2.t/ms, M_2.i, '.b')
#    
#M_1.count
#M_3.count

y_1=[]
for j in range(0,len(p_1)):
    y_1.append(sum(p_1[j]))
    
y_2=[]
for j in range(0,len(p_2)):
    y_2.append(sum(p_2[j]))
    
plot(k,y_1)
figure()
plot(k,y_2)
