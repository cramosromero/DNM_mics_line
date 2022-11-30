# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 11:08:08 2022

@author: SES271@CR
tests from Fabio's code.

This script shows the Directivity semiespheres of a Drone flying-by transversaly to
a Ground Plate Microphones line (9 microphones).
References of microphone configuration: "NASA-UNWG/SG2
UAM Ground & Flight Test Measurement Protocol""
 
"""
import os
import matplotlib.pyplot as plt
import scienceplots #https://www.reddit.com/r/learnpython/comments/ila9xp/nice_plots_for_scientific_papers_theses_and/
plt.style.use(['science', 'grid'])

from cycler import cycler
line_cycler     = cycler(linestyle = ['-', '--', '-.', (0, (3, 1, 1, 1)), ':'])
color_cycler    = cycler(color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5'])
line_color_cycler = (len(line_cycler)*color_cycler
                     + len(color_cycler)*line_cycler)
lc = line_color_cycler()
# import pandas as pd
from matplotlib import rc
# from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.cbook import get_sample_data
from matplotlib.image import imread
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# ## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import numpy as np 
import FileTools
import TimeTools
import FreqTools
import AircraftTools

plt.close('all')

# %% Constants
# #############
n_mics  = 9
to_bands =[20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250,
        315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
        5000, 6300, 8000, 10000, 12500, 16000, 20000]
# %%  Noise RECORDINGS .mat files
# ################################

# FIELD MEASUREMENTS 
PILOT   = 'Ed'
DroneID = 'M3' #{M3, 3p, Yn,Fp}
HAGL    = 10
OPE     = 'F15' #{F15, F05, F27}
PYL     = 'N' #{Y, N}
STA     = 'W' #{E, W}
Date    = '??????'

Count   = 'dw' #{uw: upwind, dw:  downwind}

identifier, files = FileTools.list_files (PILOT,DroneID,HAGL,OPE,PYL,STA,Date,Count)

import os
ffolder = 'Figs_'+identifier
if not os.path.exists(ffolder):
    os.mkdir(ffolder)

im_drone_file = os.getcwd() +"\\UASimg\\"+ DroneID+PYL+'_front.png' #"drone_N.png"
im_drone_file_over = os.getcwd() +"\\UASimg\\"+ DroneID+PYL+'_over.png'#"drone_Q_over.png"

print(identifier)
fly_speed = float(OPE[-2:])

# ACCCES DATA_ACU SINGLE EVENT
event   = 3 # {1,2,3,4,5,6,7,8,9... nEve}
acu_metric = 'LAFp' # LAFp, LZfp

Fs, TT, DATA_raw, DATA_acu = FileTools.data_single_events (files[event-1], n_mics, acu_metric)

# %%%% PLOTS all microphones, single events
# ################################
"""PLOTS all microphones, single events"""
fig, (ax0) = plt.subplots(figsize=(6,3))

for mics in range(DATA_raw.shape[1]):
    plt.plot(TT, DATA_raw[:,mics],**next(lc),linewidth=0.5,label='Ch {}'.format(mics+1))

plt.title('Flyby {}'.format(event))
plt.legend(ncol=3, loc= 8)
ax0.get_legend().set_title("Microphone")
plt.ylabel('Pressure [Pa]')
plt.xlabel('Time [s]')
plt.tight_layout()
plt.savefig(ffolder+'/'+identifier+'_'+'allChan_Press'+'Flyby{}'.format(event)+'.jpg', format='jpg',dpi=1200)
# plt.show()

# %% DATA_RAW and DATA_ACU ALL events 1-mic
# ##################################################
Fs,TT, DATA_raw_events, DATA_acu_events = FileTools.data_all_events (files, DATA_raw, n_mics, acu_metric)

# %%A-weighting (Optional)
# ################################

mic_ID = np.arange(DATA_raw_events.shape[2])+1
eve_ID = np.arange(DATA_raw_events.shape[0])+1

microphone = 5 # {1,2,3,4,5,6,7,8,9}

DATA_mic = DATA_raw_events[:,:,microphone-1].T #[event, time, mic]


# %%%% PLOTS DATA_mic same microphone, all events
"""PLOTS DATA_mic same microphone, all events"""
fig, (ax0) = plt.subplots(figsize=(6,3))

for eve in range(DATA_mic.shape[1]):
    plt.plot(TT, DATA_mic[:,eve],**next(lc),linewidth=1,label='{}'.format(eve+1))
plt.title('Microphone {}'.format(microphone))    
plt.legend(ncol=3, loc= 8)
ax0.get_legend().set_title("Flybys")
plt.ylabel('Pressure [Pa]')
plt.xlabel('Time [s]')
plt.tight_layout()
plt.savefig(ffolder+'/'+identifier+'_'+'allFlybys_Press'+'Chan{}'.format(event)+'.jpg', format='jpg',dpi=1200)
# plt.show()

# %% SIGNAL SEGMENTATION 
##############################
""" SIGNAL SEGMENTATION based on LA_max value of the medianfilter signal
    This segment could be analized with FFT to obtain the PSD, then the band content"""
    
segmented_based = 'time' #{'time','level'}   

""" Based on time slice with max value centered"""
time_slice = 1.5

TIME_r, vec_time, Data_raw_segmented, Data_acu_segmented = TimeTools.segment_time_ADJUST (time_slice, Fs,
                                                                                          DATA_raw_events, DATA_acu_events)
# %%%% PLOT segmented by time
"""PLOTS DATA_mic same microphone, all events"""
fig, (ax0) = plt.subplots(figsize=(6,3))
for eve in range(Data_raw_segmented.shape[0]):
    plt.plot(TIME_r, Data_acu_segmented[eve,:,microphone-1],**next(lc),linewidth=1,label='{}'.format(eve+1))
plt.title('Microphone {}'.format(microphone))    
plt.legend(ncol=3, loc= 8)
ax0.get_legend().set_title("Flybys")
plt.ylabel(' Amplitude [Pa]')
plt.xlabel('Time [s]')
plt.tight_layout()
# plt.savefig(ffolder+'/'+identifier+'_'+'allFlybys_Press'+'Flyby{}'.format(event)+'.jpg', format='jpg',dpi=1200)
# plt.show()
    
# %%%% If data selectio depends on threshold 
if segmented_based == 'time':
    pass

elif segmented_based == 'level': 
    """ Based on time signals lower than the Lmax"""
    THRESH = 20
    TIME_r, vec_time, Data_raw_segmented_trh, Data_acu_segmented_trh  = TimeTools.segment_THRESH(THRESH,Fs,Data_raw_segmented, Data_acu_segmented)
       
    ## PLOT segmented by threshold
    """PLOTS DATA_mic same microphone, all events"""
    fig, (ax0) = plt.subplots(figsize=(6,3))
    for eve in range(Data_raw_segmented.shape[0]):
        plt.plot(Data_acu_segmented_trh[eve,:,microphone-1],**next(lc),linewidth=1,label='{}'.format(eve))
    plt.title('Microphone {}'.format(microphone))    
    plt.legend(loc='upper right', borderpad=0.1,frameon=True)
    ax0.get_legend().set_title("Events")
    plt.ylabel(' Amplitude [Pa]')
    plt.xlabel('Time [s]')
    plt.show()
    
    Data_raw_segmented = Data_raw_segmented_trh
    Data_acu_segmented = Data_acu_segmented_trh

# %% PSD calculation
######################
Ndft    = 2**16#2**14
# Ndft = Data_raw_segmented.shape[1]*2
df      = Fs/Ndft
p_ref   = 20e-6
NOVERLAP = 2**12
WINDOW = 'hann'
# freq    = np.linspace(0, Fs - df, Ndft)[:Ndft//2+1]

"""PSD SEGMENTED same microphone, all events"""
DATA_PSD_events, freq = FreqTools.calc_PSDs(Data_raw_segmented,Fs, Ndft, WINDOW)#, NOVERLAP)
freq = np.array(freq)

# %% PSD PLOTS
###################### 
"""PLOTS PSD same microphone, all events"""
fig, axs = plt.subplots(DATA_PSD_events.shape[0], 1,figsize=(10,5.3),sharex=True)

for eve in range(DATA_PSD_events.shape[0]):
    axs[eve].plot(freq, 10*np.log10(DATA_PSD_events[eve,:,microphone-1]/p_ref**2),**next(lc),linewidth=1,label='{}'.format(eve+1))
    axs[eve].set_xscale('log')
    axs[eve].grid(linewidth=0.8)
    axs[eve].grid(b=True, which='minor', color='gray', linestyle='--',linewidth=0.5)
    axs[eve].legend(ncol=3, loc= 0)
    axs[eve].set_xlim([10, 10000])
    axs[eve].set_ylim([5,70])
    axs[eve].get_legend().set_title('Flyby')
    

fig.suptitle('Microphone {}'.format(microphone))
fig.supxlabel('Frequency [Hz]')
fig.supylabel('Amplitude [dB/Hz] re $(20 uPa)^2$')
plt.tight_layout()
plt.savefig(ffolder+'/'+identifier+'_'+'Flybys_PSD'+'Mic{}'.format(microphone)+'.jpg', format='jpg',dpi=1200)

# plt.show()

fig, (ax0) = plt.subplots(1, figsize=(6.7,3.8))

for eve in range(DATA_PSD_events.shape[0]):
    ax0.plot(freq, 10*np.log10(DATA_PSD_events[eve,:,microphone-1]/p_ref**2),**next(lc),linewidth=1,label='{}'.format(eve+1))
   
ax0.set_xscale('log')
ax0.grid(linewidth=0.8)
ax0.grid(b=True, which='minor', color='gray', linestyle='--',linewidth=0.5)
ax0.set_xlim([10, 10000])
ax0.set_ylim([5,70])
ax0.legend(ncol=3, loc= 0)
ax0.get_legend().set_title('Flyby')

   
ax0.set_ylabel('Amplitude [dB/Hz] re $(20 uPa)^2$')
ax0.set_xlabel('Frequency [Hz]')

ax0.set_title('Microphone {}'.format(microphone)) 
plt.tight_layout()
plt.savefig(ffolder+'/'+identifier+'_'+'Flybys_PSD_all'+'Mic{}'.format(microphone)+'.jpg', format='jpg',dpi=1200)
# plt.show()

#
"""PSD same microphone, all Events MEAN and STD"""
PSD_MIC_DB  = 10*np.log10(DATA_PSD_events[:,:,microphone-1]/p_ref**2)
MIC_DB_mean = np.mean(PSD_MIC_DB,axis=0)
MIC_DB_std  = np.std(PSD_MIC_DB,axis=0)

"""PLOTS PSD same microphone, all Events MEAN and STD / 
SECECTION OF FREQUENCY RANGE"""
fig, (ax0) = plt.subplots(figsize=(6,3))#,gridspec_kw={'height_ratios':[10,1]})

ax0.plot(freq,MIC_DB_mean,label='Average')#'g-'
ax0.set_xscale('log')
ax0.fill_between(freq, MIC_DB_mean - MIC_DB_std,
                  MIC_DB_mean + MIC_DB_std,alpha=0.4,label='$\pm$ std')# color='gray', 
ax0.grid(linewidth=0.8)
ax0.grid(b=True, which='minor', color='gray', linestyle='--',linewidth=0.5)
ax0.set_xlim([10, 20000])
ax0.set_ylim([5,70])
ax0.set_ylabel(' Amplitude $[dB/Hz]$ re $(20 \mu Pa)^2$')
ax0.set_xlabel('Frequency $[Hz]$')
ax0.set_title('Microphone {}'.format(microphone)) 
plt.tight_layout()

# line1, = ax1.plot([], [])
plt.legend()
plt.tight_layout()
plt.savefig(ffolder+'/'+identifier+'_'+'Mean_PSD_all'+'Mic{}'.format(microphone)+'.jpg', format='jpg',dpi=1200)
# plt.show()

# %%Spectrogram
# ################################
"""PLOT-SPECTROGRAM"""  
import scipy.signal as ss 
#eve_mic = [int, int]
eve_mic = [1, 5]

x = DATA_raw_events[eve_mic[0]-1,:,eve_mic[1]-1]

fig, (ax0) = plt.subplots(figsize=(6,3))
f, t, Sxx = ss.spectrogram(x, nperseg=(2**18)//8, fs=Fs, nfft=2**18)

plt.pcolormesh(t, f, 10*np.log10(Sxx/(20e-6)**2), cmap='jet', vmin=5,vmax=60, shading='auto')
plt.title('Spectrogram of Flyby {}'.format(eve_mic[0]))
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.ylim(100,500)
cbar = plt.colorbar()
cbar.set_label("SPL [dB/Hz]")
plt.tight_layout()
plt.savefig(ffolder+'/'+identifier+'_'+'SpectA'+'.jpg', format='jpg',dpi=1200)
# plt.show()

fig, (ax0) = plt.subplots(figsize=(6,3))
spectrum = plt.specgram(x, NFFT=2**17, Fs=Fs,cmap='jet_r') #, noverlap=2**5-1
plt.title('Spectrogram of Flyby {}'.format(eve_mic[0]))
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.ylim(100,500)
cbar = plt.colorbar()
cbar.set_label("Power/frequency [dB/Hz]")
plt.tight_layout()
plt.savefig(ffolder+'/'+identifier+'_'+'SpectB'+'.jpg', format='jpg',dpi=1200)
# plt.show()


#%% BAND Selection for Directivity
####################################

""" Selecciion de SPL ALL EVENTS on EACH MICROPHONE"""
band_selection='oto_band' #{'we_band', 'oto_band'}

# %%%% whichever band
#####################
"""IF I select wichever band"""
# freqs_selected = line1._x # from plot selection if is avaliable
# A = freq >= freqs_selected[0]
# B = freq <= freqs_selected[-1]
if band_selection == 'we_band':

    fd_ = 100
    fu_ = 400
    band = (fu_+fd_)//2
    
    A = freq >= fd_
    B = freq <= fu_
    C = np.logical_and(A,B)
    freqs_selected = freq[C]
    p2_selected = DATA_PSD_events[:,C,:] # SIEMENS CORRECTION.
    p2_sum_selected = np.sum(p2_selected,axis=1) #array [event, channles] 
    SPL_event_mic = 10*np.log10(p2_sum_selected/(p_ref**2)) 
    
    SPL_event_mic = SPL_event_mic.reshape(DATA_raw_events.shape[0],1, DATA_raw_events.shape[2]) #[eve,SPL,channel]
    label = '$SPL[dB]$ at band {} Hz'.format(band)
# %%%% among 1/3octave band or overall level
#####################
elif band_selection == 'oto_band':
    
    """IF I select wichever band"""
    fraction = 3
    DATA_SPLoverall, DATA_SPLs, freq = FreqTools.calc_SPLs(Data_raw_segmented,Fs, fraction, limits=[20, 20000], show=0)
    
    band = 200
    
    """ [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250,
            315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
            5000, 6300, 8000, 10000, 12500, 16000, 20000, 'overall'] """
    
    if type(band) == int:
        b_index = to_bands.index(band)
        
        spl_s_mean = np.mean(DATA_SPLs[:,:,:],axis=0)
        spl_s_std = np.std(DATA_SPLs[:,:,:],axis=0)
        
        SPL_event_mic = DATA_SPLs[:,b_index,:] # data [events]
        SPL_event_mic = SPL_event_mic.reshape(DATA_raw_events.shape[0],1, DATA_raw_events.shape[2]) #[eve,SPL,channel]
        label = '$SPL[dB]$ at band {} Hz'.format(band)
    
    elif (type(band) == str and band =='overall') == True: ## to calculate the overall SPL from bands
        b_index = to_bands.index(1000)
        band = 1000 #for atmospheric correction
        
        spl_s_mean = np.mean(DATA_SPLoverall[:,:,:],axis=0)
        spl_s_std = np.std(DATA_SPLoverall[:,:,:],axis=0)
        
        SPL_event_mic = DATA_SPLoverall[:,:,:] # data [events]
        label = '$SPL[dB]$ Overall'
        

# %% DEPROPAGATION
""" SPL DEPROPAGATION"""

rad_to_deprop = 1 #[m]
heightAGL = HAGL #[m]

Lmax_array = SPL_event_mic

L_array_depropagated, thetas, dist_ground_mics, phi_dx = AircraftTools.depropagator(Lmax_array, heightAGL, rad_to_deprop, band, dx=0)

theta_ref = np.max(np.array(thetas))
theta_ref_radians = (theta_ref/180)*np.pi

L_array_deprop_mean = np.squeeze(np.mean(L_array_depropagated, axis=0))
L_array_deprop_std = np.squeeze(np.std(L_array_depropagated, axis=0))

DI_events = AircraftTools.directivity (L_array_depropagated)

DI_mean = np.squeeze(np.mean(DI_events, axis=0))
DI_std = np.squeeze(np.std(DI_events, axis=0))

"""Levels After Depropagation"""
LEVEL_TO_PLOT = L_array_deprop_mean
STD_TO_PLOT = L_array_deprop_std
    
#%%%% PLOTS DIRECTIVITY-LEVEL mean std, all events - all mics
#########################
"""PLOTS DIRECTIVITY-LEVEL mean std, all events - all mics"""
band = str(band)
dfor = next(lc) #draw_format
dfor = dfor["color"] + dfor["linestyle"]

from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.cbook import get_sample_data
from matplotlib.image import imread

theta_ref = np.max(np.array(thetas))
theta_ref_radians = (theta_ref/180)*np.pi

theta = np.linspace((np.pi/2)-theta_ref_radians, (np.pi/2)+theta_ref_radians, len(thetas))

fig = plt.figure(figsize=(3.9,3.3))
ax = fig.add_subplot(111, polar=True)
ax.plot(theta, LEVEL_TO_PLOT, dfor, label=label)
ax.legend(loc='lower center')

plt.fill_between(theta, LEVEL_TO_PLOT - L_array_deprop_std,
                  LEVEL_TO_PLOT + L_array_deprop_std,
                  color='gray', alpha=0.3)

r_min = TimeTools.round_up(np.min(LEVEL_TO_PLOT),10) - 15
r_max = TimeTools.round_up(np.max(LEVEL_TO_PLOT),10)
r_r = r_min-5

ax.set_rmin(r_min)
ax.set_rmax(r_max)
ax.set_rorigin(r_r)

ax.set_rgrids(np.linspace(r_min,r_max,int((r_max-r_min)/5+1)))

ax.set_thetamin(30)
ax.set_thetamax(150)
thetas_t = [str(angle) + '\N{DEGREE SIGN}' for angle in thetas] #thetas labels
ax.set_xticks(((np.arange(len(thetas_t))*15)+30)*np.pi/180)

ax.set_xticklabels(thetas_t)

ax.set_title('Directivity pattern'+'\n'+
              'Drone: ' + DroneID +'\n'+ acu_metric +
              ' $r=' + str(rad_to_deprop)+'[m]$')
plt.tight_layout()

""" Add drone imagen """
im = imread(get_sample_data(im_drone_file, asfileobj=False))
oi = OffsetImage(im, zoom=0.15)
ab = AnnotationBbox(oi, xy=(np.pi*90/180, r_min), frameon=False, clip_on=False, xybox=(90,r_r))#, box_alignment=(1,1))
plt.gca().add_artist(ab)

ax.set_theta_offset(np.pi) #invert the plot in vertical axis
ax.yaxis.grid(which='minor', color='#ECECEC', linestyle='--', linewidth=0.8)
plt.minorticks_on()
plt.savefig(ffolder+'/'+identifier+'_'+'Dir'+label+'.jpg', format='jpg',dpi=1200)

# plt.show()

#%%%% PLOTS DIRECTIVITY-INDEX mean std, all events - all mics
#############################
"""PLOTS DIRECTIVITY-INDEX mean std, all events - all mics"""

fig = plt.figure(figsize=(3.9,3.3))

ax = fig.add_subplot(111, polar=True)
ax.plot(theta, DI_mean, dfor, label=label)
ax.legend(loc='lower center')

plt.fill_between(theta, DI_mean - DI_std,
                  DI_mean + DI_std,
                  color='gray', alpha=0.2)

r_min = TimeTools.round_up(np.min(DI_mean),3) - 6
r_max = TimeTools.round_up(np.max(DI_mean),3)
r_r = r_min-3

ax.set_rmin(r_min)
ax.set_rmax(r_max)
ax.set_rorigin(r_r)

ax.set_rgrids(np.linspace(r_min,r_max,int((r_max-r_min)/3+1)))

ax.set_thetamin(30)
ax.set_thetamax(150)
thetas_t = [str(angle) + '\N{DEGREE SIGN}' for angle in thetas] #thetas labels
ax.set_xticks(((np.arange(len(thetas_t))*15)+30)*np.pi/180)

ax.set_xticklabels(thetas_t)

ax.set_title('DI'+'\n'+
              'Drone: '+ DroneID +'\n'+
              ' $r=' + str(rad_to_deprop)+'[m]$')
plt.tight_layout()
# ax.minorticks_on()
""" Add drone imagen """
im = imread(get_sample_data(im_drone_file, asfileobj=False))
oi = OffsetImage(im, zoom=0.15)
ab = AnnotationBbox(oi, xy=(np.pi*90/180, r_min),frameon=False, clip_on=False,xybox=(90,r_r))#, box_alignment=(1,1))
plt.gca().add_artist(ab)

ax.set_theta_offset(np.pi) #invert the plot in vertical axis
ax.yaxis.grid(which='minor', color='#ECECEC', linestyle='--', linewidth=0.8)
plt.minorticks_on()
plt.savefig(ffolder+'/'+identifier+'_'+'DI'+label+'.jpg', format='jpg',dpi=1200)
# plt.show()
#####################    
#%%RESULTS
######################
print('results')
print(identifier)