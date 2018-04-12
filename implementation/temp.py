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

I_s = 460*pA
#I_d = 200*pA;
n = 100 ; p= [] ; k =[]

for i in range (0,1000,100):
    
    I_d = i*pA
        
    eqs='''
    dv_s/dt = ( -(v_s-el)/tau_s ) + ( g_s * (1/(1+exp(-(v_d-ed)/dd))) + I_s + w)/C_s : volt (unless refractory)
    dw/dt = -w/tauw_s :amp
    dv_d/dt =  ( -(v_d-el)/tau_d ) + ( g_d*(1/(1+exp(-(v_d-ed)/dd))) +c_d*K + I_d + ws )/C_d : volt 
    dws/dt = ( -ws + aw_d *(v_d - el))/tauw_d :amp
    K :1
    '''   
    
    S = NeuronGroup(n, model=eqs,  threshold='v_s > -50*mV', reset='v_s =el ; w+= bw_s ',refractory=3*ms,method='euler') #+delay 
    
    S.v_s = el
    S.v_d = el
    
    ##############################################
    
    backprop = Synapses(S, S, 
                         on_pre={'up': 'K += 1', 'down': 'K -=1'}, 
                         delay={'up': 0.5*ms, 'down': 2.5*ms}) 
    backprop.connect()
    
    
    ##############################################
    
    soma = StateMonitor(S,'v_s',record=True)
    dend = StateMonitor(S,'v_d',record=True)
    
    kent = StateMonitor(S,'K',record=True)
    
    ##############################################
    M = SpikeMonitor(S)
    
    ####################################################################
    run(simtime)
    p.append(M.count/simtime)
    k.append(i)

def column(matrix, i):
    return [row[i] for row in matrix]

plot(k,column(p, 1)) 
xlabel('I_d (pA)',fontsize=20) ; ylabel('Firing rate (spikes per second)',fontsize=20); 
title( 'population firing rate as a function of I_d ', fontsize=20)
#
#trial = range(1,size(k)+1); plot(trial, k)
#xlabel('trial',fontsize=20) ; ylabel('I_d (pA)',fontsize=20); 
#title( ' ', fontsize=20)

#plot(i_s,p)
#xlabel('I_s (pA)',fontsize=20) ; ylabel('Firing rate (spikes per second)',fontsize=20); 
#title( 'Firing rate as I_s increases, I_d = 0 ', fontsize=20)

###################################################################
# somatic current required to generate APs
# plot of Firing rate as I_s increases, I_d = 0 

#I_d = 0*pA
#i_s = []
#for i in range(0,1000,10):
#    I_s = i*pA 
#    run(simtime)
#    p.append(M.count/simtime)
#    i_s.append(I_s)
#
#plot(i_s,p)
#xlabel('I_s (pA)',fontsize=20) ; ylabel('Firing rate (spikes per second)',fontsize=20); 
#title( 'Firing rate as I_s increases, I_d = 0 ', fontsize=20)

# >> min I_s = 460pA

##################################################################

#I_s = 460*pA
#i_d = []
#for i in range(0,1000,10):
#    I_d = i*pA 
#    run(simtime)
#    p.append(M.count/simtime)
#    i_d.append(I_d)
#
#plot(i_d,p)
#xlabel('I_d (pA)',fontsize=20) ; ylabel('Firing rate (spikes per second)',fontsize=20); 
#title( 'Firing rate as I_d increases, I_s = 460*pA ', fontsize=20)


###################################################################
#run(simtime)
#subplot(2,1,1)
#plot( soma.t/ms, soma.v_s[0], label='soma')
#plot( dend.t/ms, dend.v_d[0], label='dendrite')
#xlabel('Time (ms)',fontsize=20) ; ylabel('V (mV)',fontsize=20); 
#legend(loc='best'); title( 'n = 1 , I_d = 200pA',fontsize=20)
#    
#subplot(2,1,2)
#plot(M.t,np.ones(size(M.i)) ,'*')


#########################################################################
