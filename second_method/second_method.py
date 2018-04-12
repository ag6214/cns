# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 10:30:41 2018

@author: Windows
"""
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from brian2 import *
from brian2.units.allunits import *

start_scope()

simtime = 1*second # Simulation time

ed = -38*mV         #controls the position of the threshold
dd = 6*mV           #controls the sharpness of the threshold 
#f = 1/(1+exp(-(u-ed)/dd))

C_s = 370*pF ; C_d = 170*pF      #capacitance 
el = -70*mV                                              #reversal potential 
c_d = 2600*pA
aw_s = aw_a = aw_b = 0 ; aw_d = -13*nS                   #strength of subthreshold coupling 
bw_s = -200*pA ; bw_d = 0 ; bw_a = bw_b = -150*pA        #strength of spike-triggered adaptation 
tauw_s = 100*ms ; tauw_d = 30*ms       #time scale of the recovery variable 
tau_s = 16*ms ; tau_d = 7*ms; tau_a = 20*ms; tau_b = 10*ms # time scale of the membrane potential 

g_s = 1300*pA ; g_d = 1200*pA #models the regenrative activity in the dendrites 

I_s = 500*pA ; I_d = 0*pA #step current enough i to get soma spiking 
n = 1

#g_i = 1*nsiemens

eqs='''
dv_s/dt = ( -(v_s-el)/tau_s ) + ( g_s * (1/(1+exp(-(v_d-ed)/dd))) + I_s + w)/C_s : volt (unless refractory)
dw/dt = -w/tauw_s :amp
dv_d/dt =  ( -(v_d-el)/tau_d ) + ( g_d*(1/(1+exp(-(v_d-ed)/dd))) +c_d*K + I_d + ws + g_i*(-(v_d-el)) )/C_d  : volt 
dws/dt = ( -ws + aw_d *(v_d - el))/tauw_d :amp
K :1
g_i : siemens
'''   

P = PoissonGroup(1, 10*Hz)

S = NeuronGroup(1, model=eqs,  threshold='v_s > -50*mV', reset='v_s =el ; w+= bw_s ',refractory=3*ms,method='euler') #+delay 

S.v_s = el
S.v_d = el

##############################################

backprop = Synapses(S, S, 
                     on_pre={'up': 'K += 1', 'down': 'K -=1'}, 
                     delay={'up': 0.5*ms, 'down': 2.5*ms}) 
backprop.connect()

Y = Synapses(P, S, on_pre='g_i+=10*nsiemens')
Y.connect(j='i')



##############################################

soma = StateMonitor(S,'v_s',record=True)
dend = StateMonitor(S,'v_d',record=True)

kent = StateMonitor(S,'K',record=True)

##############################################
M = SpikeMonitor(S)
N = SpikeMonitor(P)

####################################################################

run(simtime)


#subplot(3, 1 , k)

#p.append(M.count/simtime)

#plot(list(range(0,3)),p)

#plot([ 0, 100,200 , 300, 400 ,420,450,470, 500,520, 550,570,600,650]  ,p)




plot( soma.t/ms, soma.v_s[0], label='soma')
xlabel('Time(ms)') ; ylabel('v'); legend(loc='best') ; 
plot( dend.t/ms, dend.v_d[0], label='dendrite')
xlabel('Time (ms)') ; ylabel('V (mV)'); legend(loc='best'); 






#k=k+1
#########################################################################


