# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:20:20 2023

@author: SES271
TO Create wave files from Dewesoft recordings.
"""
import numpy as np
from scipy import io #READ MAT FILES
import glob # STRINGS
import matplotlib.pyplot as plt
from cycler import cycler
line_cycler     = cycler(linestyle = ['-', '--', '-.', (0, (3, 1, 1, 1)), ':'])
color_cycler    = cycler(color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5'])
marker_cycler    = cycler(marker = ['d', '*', '^', 's', 'X', '2'])
line_color_cycler = (len(line_cycler)*color_cycler
                     + len(color_cycler)*line_cycler)
lc = line_color_cycler()
import FileTools
# %%% Constants
# ###
p_ref   = 20e-6
n_mics  = 9

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
ffolder = 'WAVS_'
if not os.path.exists(ffolder):
    os.mkdir(ffolder)
    
print(identifier)

event   = 3 # {1,2,3,4,5,6,7,8,9... nEve}
acu_metric = 'LAFp' # LAFp, LZfp

Fs, TT, DATA_raw, DATA_acu = FileTools.data_single_events (files[event-1], n_mics, acu_metric)

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
# plt.show()

# %%%  Write WAV files
#########
import soundfile as sf
for eve in range (DATA_raw_events.shape[0]):
    for mm in range(n_mics):
        eve_list = eve
        ch_list = mm
    
        mic_data = DATA_raw_events[eve_list,:,ch_list]
        sf.write(ffolder+'/'+identifier+"_ev"+str(eve_list+1)+"_M"+str(ch_list+1)+'.wav', mic_data, Fs,subtype='FLOAT')

# %%FOR CALIBRATION WAVS
# Write WAV files CALIBRATION
#########
channel = 10 
mat_folder = 'C:/Users/ses271/OneDrive - University of Salford/Documents/ARC_Salford/DroneNoiseMeas/Dewe_data/calibrationSIGNALS'

file = 'Calib_Scotland_ch'+str(channel)
data_ID_cal = mat_folder + "/" + file+".mat"

DATA = io.loadmat(data_ID_cal) #Read the .mat file
key = "Data1"+"_Mic_"+str(channel)

vector_wav=DATA[key]

import os
ffolder = 'WAVS_'
if not os.path.exists(ffolder):
    os.mkdir(ffolder)
sf.write(ffolder+'/'+file+'.wav', vector_wav, Fs,subtype='FLOAT')