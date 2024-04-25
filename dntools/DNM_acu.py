# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 11:08:08 2022

@author: SES271@CR
tests from Fabio's code.
"""
import random
import matplotlib.pyplot as plt
import scienceplots #https://www.reddit.com/r/learnpython/comments/ila9xp/nice_plots_for_scientific_papers_theses_and/
plt.style.use(['science', 'grid'])
# plt.style.use('seaborn-whitegrid')
# plt.style.use('classic')

from cycler import cycler
line_cycler     = cycler(linestyle = ['-', '--', '-.', (0, (3, 1, 1, 1)), ':'])
color_cycler    = cycler(color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5'])
marker_cycler    = cycler(marker = ['d', '*', '^', 's', 'X', '2'])
line_color_cycler = (len(line_cycler)*color_cycler
                     + len(color_cycler)*line_cycler)
lc = line_color_cycler()

# import pandas as pd
from matplotlib import rc
# from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# ## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import numpy as np 
import FileTools
import TimeTools

plt.close('all')

# %%% Constants
# ###
p_ref   = 20e-6
n_mics  = 9
# #######
## %%  Noise RECORDINGS .mat files
# ################################
linestyle=random.choice(['-', '--', '-.', (0, (3, 1, 1, 1)), ':'])
# FIELD MEASUREMENTS 
PILOT   = 'Ed'
DroneID = 'Yn' #{M3, 3p, Yn,Fp}
HAGL    = 10
OPE     = 'F15' #{F15, F05, F27}
PYL     = 'Y' #{Y, N}
STA     = 'W' #{E, W}
Date    = '??????' #{hhmmss}

Count   = 'dw' #{uw: upwind, dw:  downwind}
identifier, files = FileTools.list_files (PILOT,DroneID,HAGL,OPE,PYL,STA,Date,Count)

# %%% Folder for save the figures
#########
import os
ffolder = 'Figs_'+identifier
if not os.path.exists(ffolder):
    os.mkdir(ffolder)
    
print(identifier)
fly_speed = float(OPE[-2:])

# %%%  ACCCES DATA_ACU SINGLE EVENT
#########
event   = 3 # {1,2,3,4,5,6,7,8,9... nEve}
acu_metric = 'LAFp' # LAFp, LZfp

Fs, TT, DATA_raw, DATA_acu = FileTools.data_single_events (files[event-1], n_mics, acu_metric)

# %%%% Single event 
"""PLOTS all microphones, single events"""
fig, (ax0) = plt.subplots(figsize=(6,3))

for mics in range(DATA_acu.shape[1]):
    plt.plot(TT, DATA_acu[:,mics],**next(lc),linewidth=0.5,label='Ch {}'.format(mics+1))

plt.title('Flyby {}'.format(event))
plt.legend(ncol=3, loc= 8)#frameon=True)
ax0.get_legend().set_title("Microphone")
plt.ylabel(acu_metric + ' [dB] re $20\mu$ Pa')
plt.xlabel('Time [s]')
plt.tight_layout()
plt.savefig(ffolder+'/'+identifier+'_'+'allChan_'+'Flyby{}'.format(event)+'.jpg', format='jpg',dpi=1200)
plt.show()

# %%%  Write WAV files
#########
import soundfile as sf
ch_list = 5
mic_data = DATA_raw[:,ch_list-1]
sf.write(ffolder+'/'+identifier+"_M"+str(ch_list)+'.wav', mic_data, Fs,subtype='FLOAT')

# %% ACCCES DATA_ACU ALL EVENTS ONE MICROPHONE
Fs,TT, DATA_raw_events, DATA_acu_events = FileTools.data_all_events (files, DATA_raw, n_mics, acu_metric)
mic_ID = np.linspace(0,DATA_acu_events.shape[2]-1,DATA_acu_events.shape[2])+1
eve_ID = np.linspace(0,DATA_acu_events.shape[0]-1,DATA_acu_events.shape[0])+1

# %%%% Máximos
# Lmax_eve_chann = []
# for e in range(DATA_acu_events.shape[0]):
#     Lmax_mics = np.max(DATA_acu_events[e,:,:],axis=0)
#     Lmax_eve_chann.append(Lmax_mics)
# Lmax_eve_chann = np.array(Lmax_eve_chann)

# %%%% Single microphpne
"""PLOTS same microphones, all events"""
microphone = 5 # {1,2,3,4,5,6,7,8,9}
DATA_mic = DATA_acu_events[:,:,microphone-1].T #[event, time, mic]

fig, (ax0) = plt.subplots(figsize=(6,3))
# plt.plot(TT,DATA_mic)
for eve in range(DATA_mic.shape[1]):
    plt.plot(TT, DATA_mic[:,eve],**next(lc),linewidth=0.5,label='{}'.format(eve+1))
plt.title('Microphone {}'.format(microphone))    
plt.legend(ncol=3, loc= 8)#frameon=True)
ax0.get_legend().set_title("Flybys")
plt.ylabel(acu_metric + ' [dB] re $20\mu$ Pa')
plt.xlabel('Time [s]')
plt.tight_layout()
plt.savefig(ffolder+'/'+identifier+'_'+'allFlybys_'+'Chan{}'.format(microphone)+'.jpg', format='jpg',dpi=1200)
# plt.show()

# %%
#################################################
""" FOR FLYOVERS time relatives figs """ 
t_data = 7 #in seconds before and after of the max value

DATA_mic_relative, TIME_r = TimeTools.relatived_time (DATA_mic, Fs,t_data)
descriptive = TimeTools.descriptive_time (DATA_mic_relative, 1) #statistics
"""PLOTS same microphones, all events - relative time"""
fig, (ax0) = plt.subplots(figsize=(6,3))
for eve in range(DATA_mic_relative.shape[1]):
    plt.plot(TIME_r/Fs,DATA_mic_relative[:,eve],**next(lc),linewidth=0.5)
plt.title('Microphone {}'.format(microphone))
ax0.legend(eve_ID,ncol=3, loc= 8) #, frameon=True )
ax0.get_legend().set_title("Flybys")
plt.xlabel('Time relative [s]')
plt.ylabel(acu_metric + ' [dB] re $20\mu$ Pa')
plt.tight_layout()
plt.savefig(ffolder+'/'+identifier+'_'+'allFlybys_'+'Chan{}'.format(microphone)+'t_rela'+'.jpg', format='jpg',dpi=1200)
# plt.show()
# %%PLOTS mean std, all events - relative time"""
"""PLOTS mean std, all events - relative time"""
Y0 = descriptive[:,0] # mean
# from scipy.signal import medfilt
# Y0 = medfilt(Y0, 20001)
Y1 = descriptive[:,1] # std
Ymin = (np.min(descriptive[:,2])//10)*10
Ymax = np.max(descriptive[:,3])+3


fig, (ax0) = plt.subplots(figsize=(6,3))
plt.plot(TIME_r/Fs,Y0,label=' avg.' + acu_metric + '. sUAS: '+DroneID,linestyle=linestyle)#,'g-')
plt.fill_between(TIME_r/Fs, Y0 - Y1, Y0 +Y1, alpha=0.2)#,label='$\pm$ std')#,color='gray', alpha=0.2)
plt.title('Fly-bys @'+OPE[1:3]+'m/s. '+'Mic. {}'.format(microphone))
plt.legend()
plt.xlabel('Time relative [s]')
plt.ylabel(acu_metric + ' [dB] re. $20\mu$ Pa')
plt.tight_layout()
plt.savefig(ffolder+'/'+identifier+'Arglevel_@Chan{}'.format(microphone)+'t_rela'+'.jpg', format='jpg',dpi=1200)

# plt.show()

# # %% SEL calculation from ALL EVENTS ALL MICROPHONE
# #####################################################
""" FOR FLYOVERS Lmax & SEL CALCULATION by MICROPHONE"""
Lmax_array, SEL_k_array, SEL_def_array, Time_sel = TimeTools.SEL_calc(DATA_acu_events, Fs)

Lmax_mean = np.mean(Lmax_array, axis=0)
Lmax_std = np.std(Lmax_array, axis=0)

""" Select the correct SEL calculation, by deffinition is: SEL_def_array"""
SEL_array = SEL_def_array
SEL_mean = np.mean(SEL_array, axis=0)
SEL_std = np.std(SEL_array, axis=0)

"""PLOTS LMAX mean std, all events - all mics"""
fig, (ax0) = plt.subplots(figsize=(6,3))
plt.errorbar(mic_ID, Lmax_mean, yerr=Lmax_std, fmt='.k', elinewidth=1, capsize=0, ecolor='g')

ax0.set_title(acu_metric)
ax0.set_xlabel("Microphone")
ax0.set_ylabel('$L_{MAX} [dB]$')
ax0.xaxis.set_ticks_position('none')
plt.tight_layout()
plt.savefig(ffolder+'/'+identifier+'LMAX_chan_alleve'+'.jpg', format='jpg',dpi=1200)
# plt.show()

"""PLOTS SEL mean std, all events - all mics"""
fig, (ax0) = plt.subplots(figsize=(6,3))
plt.errorbar(mic_ID, SEL_mean, yerr=SEL_std, fmt='.k', ecolor='g', elinewidth=1, capsize=0);
ax0.set_title('SEL')
ax0.set_xlabel("Microphone")
ax0.set_ylabel('$SEL [dB]$')
ax0.xaxis.set_ticks_position('none')
plt.tight_layout()
plt.savefig(ffolder+'/'+identifier+'SEL_chan_alleve'+'.jpg', format='jpg',dpi=1200)
# plt.show()

#%%RESULTS
######################
print('results')
print(identifier)
RR=np.array([Lmax_mean,Lmax_std,SEL_mean,SEL_std]).T
np.savetxt(ffolder+'/'+identifier+'LmaxSEL'+'.csv', RR, delimiter=",")
time_SEL = np.around(np.mean(Time_sel,axis=0)[4],1)
print(np.around(RR,1))
print(str(time_SEL)+'[s] SEL_t at central mic')
#%%Dictionary for configuration figures
######################
C_G_JAST={"identifier"  :identifier,
          "event"       :event,
          "acu_metric"  :acu_metric,
          "microphone"  :microphone,
          "t_data"      :t_data,
          "time_SEL"    :time_SEL}
import pickle #https://wiki.python.org/moin/UsingPickle
# save dictionary to pickle file
pickle.dump( C_G_JAST, open(ffolder+'/'+identifier+"_acu.p", "wb" ) )
### read
## AA = pickle.load( open( ffolder+'/'+identifier+"_acu.p", "rb" ) )