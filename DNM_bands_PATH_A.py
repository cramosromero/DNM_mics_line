# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 10:55:32 2023
@author: SES271@CR
tests from Fabio's code.

This script shows the Directivity semiespheres of a Drone flying-by transversaly to
a Ground Plate Microphones line (9 microphones).
References of microphone configuration: "NASA-UNWG/SG2
UAM Ground & Flight Test Measurement Protocol""
 
"""
# import os
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

# %%  Noise RECORDINGS .mat files
# ################################

# FIELD MEASUREMENTS 
PILOT   = 'Ed'
DroneID = 'M3' #{M3, 3p, Yn,Fp}
HAGL    = 10
OPE     = 'F15' #{F15, F05, F27}
PYL     = 'N' #{Y, N}
STA     = 'E' #{E, W}
Date    = '??????' #{hhmmss}

Count   = 'uw' #{uw: upwind, dw:  downwind}

identifier, files = FileTools.list_files (PILOT,DroneID,HAGL,OPE,PYL,STA,Date,Count)

import os
ffolder = 'Figs_'+identifier
if not os.path.exists(ffolder):
    os.mkdir(ffolder)
im_drone_file = os.getcwd() +"\\UASimg\\"+ DroneID+PYL+'_front.png' #"drone_N.png"
im_drone_file_over = os.getcwd() +"\\UASimg\\"+ DroneID+PYL+'_over1.png'#"drone_Q_over.png

# %% Constants
# #############
n_mics      = 9
to_bands    =[20,25,31,40,50, 63, 80, 100, 125, 160, 200, 250,
              315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
              5000, 6300, 8000, 10000,12500,16000,20000]
print(identifier)
fly_speed   = float(OPE[-2:])
 #total time considering the two sides from Lmax in seconds
event       = 3 # {1,2,3,4,5,6,7,8,9... nEve}
microphone  = 5 # {1,2,3,4,5,6,7,8,9... nMic}
acu_metric  = 'LAFp' # LAFp, LZfp
# %% DATA_RAW and DATA_ACU ALL events 1-mic
# ##################################################
# ACCCES DATA_ACU SINGLE EVENT
Fs, TT, DATA_raw, DATA_acu = FileTools.data_single_events (files[event-1], n_mics, acu_metric)
# ACCCES DATA_ACU ALL EVENTS
Fs, TT, DATA_raw_events, DATA_acu_events = FileTools.data_all_events (files, DATA_raw, n_mics, acu_metric)

#%%"""PLOTS all microphones, single events"""
# fig, (ax0) = plt.subplots()

# for mics in range(DATA_raw_events.shape[2]):
#     plt.plot(TT, DATA_raw_events[event-1,:,mics],**next(lc),linewidth=1,label='Ch {}'.format(mics+1))

# plt.title(DroneID + ' Event {}'.format(event))
# plt.legend(loc='upper right', borderpad=0.1,frameon=True)
# ax0.get_legend().set_title("Microphone")
# plt.ylabel(' Amplitude [Pa]')
# plt.xlabel('Time [s]')
# plt.show()

# %% PRE-SIGNAL SEGMENTATION and Time fitting based on LMAX
##############################
""" SIGNAL SEGMENTATION based on LA_max value of the medianfilter signal
    This segment could be analized with FFT to obtain the PSD, then the band content"""
time_slice  = 2.5 #in seconds
TIME_r, vec_time, Data_raw_segmented, Data_acu_segmented = TimeTools.segment_time_ADJUST(time_slice, Fs,
                                                                                          DATA_raw_events, DATA_acu_events)
#%% """PLOTS DATA_mic same microphone, all events"""
# fig, (ax0) = plt.subplots()
# for eve in range(Data_raw_segmented.shape[0]):
#     plt.plot(TIME_r, Data_acu_segmented[eve,:,microphone-1],**next(lc),linewidth=1,label='{}'.format(eve))
# plt.title(DroneID + ' Microphone {}'.format(microphone))    
# plt.legend(loc='upper right', borderpad=0.1,frameon=True)
# ax0.get_legend().set_title("Events")
# plt.ylabel(' Amplitude [dB]')
# plt.xlabel('Time [s]')
# plt.show()

# %% SIGNAL SEGMENTATION and Time fitting based on LMAX
##############################
""" Based on time signals lower than the Lmax"""
THRESH = 10 #in dB

segment_by_THRESH_events_acu, segment_by_THRESH_events_raw = TimeTools.segment_THRESH(THRESH,Fs,Data_raw_segmented, Data_acu_segmented,microphone)

"""PLOTS DATA_mic same microphone, all events"""
fig, (ax0) = plt.subplots()
for eve in range(len(segment_by_THRESH_events_acu)):
    plt.plot(segment_by_THRESH_events_acu[eve][:,microphone-1],**next(lc),linewidth=1,label='{}'.format(eve))
plt.title(DroneID + ' Microphone {}'.format(microphone))    
plt.legend(loc='upper right', borderpad=0.1,frameon=True)
ax0.get_legend().set_title("Events")
plt.ylabel(' Amplitude [dB]')
plt.xlabel('Time [s]')

# %% SEGMENT_based DIRECTIVITY
##############################
event = 3 # {1,2,3,4,5,6,7,8,9... nEve}.
segmented_based = 'time' #{'time','level'} 

if segmented_based=='time':
    lev_vec = Data_acu_segmented[event-1,:,:]
    raw_vec = Data_raw_segmented[event-1,:,:]
elif segmented_based=='level': #time_slice  > 10
    lev_vec = segment_by_THRESH_events_acu[event-1] # size(data,mics)
    raw_vec = segment_by_THRESH_events_raw[event-1] # size(data,mics)

raw_vec = np.reshape(raw_vec,(1,raw_vec.shape[0],raw_vec.shape[1])) #suitable for the SPL by ban definition
t_chunk = lev_vec.shape[0]/Fs
print("seconds of selected chunk:"+str(t_chunk))
fig, (ax0) = plt.subplots()
tvec = np.linspace(0,lev_vec[:,4].shape[0],lev_vec[:,4].shape[0])/Fs
plt.plot(tvec,lev_vec[:,4])

# %% PATH chunks 
###LAmx point as reference
################################
#find the sample with the(Lmax) at central microphone
max_Lmax_sam = np.argmax(lev_vec[:,4])
plt.plot(tvec[max_Lmax_sam],np.max(lev_vec[:,4]),marker='o')
nspls = lev_vec.shape[0] # number of samples in Section
n_ch = 11 #number of chunks eary if it is an odd number
jump = nspls//n_ch
""" MAP of segments in the fligh path"""
si = [0]
sf = [jump-1]
sm = [round(sf[0]/2)+1]

for nw in range(n_ch-1):
    si.append(sf[nw]+1)
    sf.append(sf[nw]+jump)
    sm.append(sm[nw]+jump)
""" Vector dx"""
"estimation based on the central sample"

dx = []
for dx_i in range(len(sm)):
    dx.append(fly_speed*(abs(max_Lmax_sam-sm[dx_i])/Fs)) ##norm
#correctio#
#sm[list(abs(sm-max_Lmax_sam)).index(min(abs(sm-max_Lmax_sam)))]=max_Lmax_sam

levs_by_band_chunck = [] #saving the propagated band levels on each chunk

"""SLP of each chuck of data"""
for win_ord in range(len(sm)):
    nsi = si[win_ord]       #   initial sample
    nsf = sf[win_ord]       #   final sample   
    nsc = sm[win_ord]       #   central sample
    """ 1/3 octave band"""
    fraction = 3
    DATA_SPLoverall, DATA_SPLs, freq = FreqTools.calc_SPLs(raw_vec[:,nsi:nsf,:],
                                                           Fs, fraction, limits=[20, 20000], show=0)
    levs_by_band_chunck.append(DATA_SPLs)
    
""" SPL back - PROPAGATION """
L_by_band_chunk_BP = [] #saving the propagated band levels on each chunk BAck propagated
DI_PHI = []
rad_to_deprop = 1 #[m]
heightAGL = HAGL #[m]


for win_ord in range(len(levs_by_band_chunck)):
    L_array_all_freqs = levs_by_band_chunck[win_ord]
    L_array_all_freqs_BP = np.ones((1,len(freq), n_mics))*10
    for n_band in range(len(freq)):
        band = freq[n_band]
        L_array = L_array_all_freqs[0,n_band,:]
        L_array_to_BP = np.reshape(L_array,(1,1,L_array.shape[0])) #array suitable for backpropagation
        L_array_depropagated, thetas, dist_ground_mics, phi_dx = AircraftTools.depropagator(L_array_to_BP,heightAGL, rad_to_deprop, band, dx[win_ord])
        L_array_all_freqs_BP[0,n_band,:]=L_array_depropagated
    DI_PHI.append(phi_dx)
    L_by_band_chunk_BP.append(L_array_all_freqs_BP)

#%%  BAND's SPL or OASPL from bands contributions
#########################
L_by_OASPL_chunk_BP = []

bi = to_bands.index(50) # lower band
bf = to_bands.index(10000) # upper band

for n_ch in range(len(L_by_band_chunk_BP)):
    L_by_OASPL_chunk_BP.append(10*np.log10(np.sum(10**((L_by_band_chunk_BP[n_ch][0,bi:bf+1,:])/10),axis=0)))

if bi==bf:
    label = '$SPL[dB]$ at band {} Hz'.format(to_bands[bi])
else:
    label = '$SPL[dB]$ at bands from {} Hz to {}'.format(to_bands[bi],to_bands[bf]) 
#%% PLOTING 
###########
LEVELS_th_ph = np.array(L_by_OASPL_chunk_BP)

# limits for colorplots  
plt_Level_min = np.min(LEVELS_th_ph)
plt_Level_max = np.max(LEVELS_th_ph)

"""PLOTS SEMIESPHERES, all events - all mics"""
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
## Coordinates
## Azimut angle 
th_min = 0
th_max = 180
## Polar angle
ph_min = -90
ph_max = 90

j_15 = 1 + n_mics+ (180-(n_mics*15))//15 # Mics angle, jump+ 15 deg.
th   = np.linspace(th_min*np.pi/180, th_max*np.pi/180, j_15 ) 

all_phi_deg = 90-(np.array(DI_PHI)*180/np.pi) # Path angle. coordinathes acording with NORAH
ph_neg = all_phi_deg.argmin()
ph = np.concatenate([all_phi_deg[0:ph_neg]*-1, all_phi_deg[ph_neg:]]) 
ph = ph*np.pi/180

th, ph = np.meshgrid(th,ph)
"""AQUI"""
## NORAH coordinates
x = 10 * np.cos(th)
y = -10 * np.sin(th) * np.sin(ph)
# HAGL = HAGL
z = HAGL - (10 *np.sin(th) * np.cos(ph))

## BIG semihespheres 
#  DL Directivity levels
#  DSTD Distectivity STD
DL_theta_phi = np.ones(th.shape)*(plt_Level_min-6) # 30 db as background noise
DSTD_theta_phi = np.ones(th.shape)*(-1) # 
# LEVELS LOCATION on the SEMIESPHERE
for i_phi in range(LEVELS_th_ph.shape[0]):
    DL_theta_phi[i_phi,2:11] = LEVELS_th_ph[i_phi,:]

mask = np.ones(th.shape) #zones without information
masked = np.ma.masked_where(DL_theta_phi>plt_Level_min-1, mask)#zones without information

## FROM CONTROL ROOM POINT OF VIEW
strength = np.flip(np.flip(DL_theta_phi,axis=0),axis=1) 

if Count =='dw' : ##flip the aray if the drone flyes up-wind (for the order in the Sphere)
    LEVELS_th_ph = np.flip(LEVELS_th_ph, 1)

cmap = plt.colormaps["viridis"]
norm = colors.Normalize(vmin = np.min(strength), vmax = np.max(strength), clip = False)
# Figure
fig = plt.figure(figsize=(8,8))
plt.figaspect(1)
ax = fig.add_subplot(111, projection='3d')
""" axis directivity"""
ec = ax.plot_surface(x, y, z, edgecolor='none',facecolors = cmap(norm(strength)),
                      cmap=cmap, antialiased=False)
ax.set_box_aspect([1,1,1])
# %%%%%% Mic-line as a reference
for nm in range(1, len(dist_ground_mics), 1):
    xcor = dist_ground_mics[nm]*HAGL / dist_ground_mics[0]
    if nm > len(dist_ground_mics)//2:
        xcor= xcor*-1
        
    ax.plot([xcor],[0],[-HAGL],marker='.', color='k')

if Count =='uw' :
    ax.plot([dist_ground_mics[0]*HAGL / dist_ground_mics[0]],[0],[-HAGL],marker='P', color='r')
else:
    ax.plot([-1*dist_ground_mics[0]*HAGL / dist_ground_mics[0]],[0],[-HAGL],marker='P', color='r')
  
# %%%%%% Figure axis
ax.set_axis_off()
plt.title('SPL Hemispheres'+'\n'+ DroneID+'\n'+label)
ax.set_xlabel('N-S X axis')
ax.set_ylabel('W-E Y axis')
ax.set_zlabel('Z axis')

ax.view_init(28,137)
plt.tight_layout()

plt.savefig(ffolder+'/'+identifier+'_'+'SPLHemi'+ DroneID +segmented_based+'_ev_'+str(event)+'.jpg', format='jpg',dpi=1200)
plt.show()

# %%%%%% Colorbar
"""COLORBAR"""
import matplotlib.pyplot as plt
import matplotlib as mpl

fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)

cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal',
             label ='[dB]'.format(band))
cbt = cbar.get_ticks().tolist()
cbt[0]='$no$ $data$'
cbar.ax.set_xticklabels(cbt)

plt.tight_layout()
plt.savefig(ffolder+'/'+identifier+'_'+'SPLHemi'+ DroneID +segmented_based+'_ev_'+str(event)+'cb.jpg', format='jpg',dpi=1200)
# %%%%%% "unwrapped"-directivity
import scipy.ndimage
from matplotlib.ticker import AutoMinorLocator
th_ticks_label = ['','','1','2','3','4','5','6','7','8','9','','']
deg_th = np.array([-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90])
if Count  == 'dw':
    th_ticks_label = ['','','9','8','7','6','5','4','3','4','1','','']
    # deg_th = np.flip(deg_th,0)

deg_ph = -1*ph[:,0]*180/np.pi

fig, ax = plt.subplots(figsize=(5,8))
plt.grid(False)

plt.imshow(DL_theta_phi, cmap=cmap,interpolation ='gaussian',vmin=np.round(np.min(DL_theta_phi)), vmax=np.round(np.max(DL_theta_phi)))#, interpolation ='none', alpha=0.8)
cbar = plt.colorbar(orientation='horizontal', alpha=0.8,label ='$SPL$ [dB]', fraction=0.075, pad=0.07)#, location='bottom')
plt.clim(plt_Level_min);
# cbt = cbar.get_ticks().tolist()
# cbt[0]='$no$ $data$'
# cbar.ax.set_xticklabels(cbt)
# plt.gca().invert_yaxis()
# plt.xticks()
plt.ylabel(r'$\Phi$')
plt.yticks(np.arange(0, deg_ph.shape[0], 1),np.round(deg_ph,1),fontsize=8)

plt.xlabel('Microphone')#r'$\Theta$')
plt.xticks(np.arange(0, len(th_ticks_label), 1),th_ticks_label, rotation=0,fontsize=8)
# plt.title('DI '+label)

plt.imshow(masked, 'gray', interpolation='none', alpha=0.8) #zones without information

secax = ax.secondary_xaxis('top') #second axis THETA
secax.set_xticks(np.arange(0, len(th_ticks_label), 1),deg_th, rotation=45,fontsize=8)
secax.set_xlabel('$\Theta$')
plt.tight_layout()
""" Add drone imagen """
im = imread(get_sample_data(im_drone_file_over, asfileobj=False))
oi = OffsetImage(im, zoom=0.03)
ab = AnnotationBbox(oi, xy=(len(th_ticks_label)//2,all_phi_deg.argmin()), frameon=False, clip_on=False)#, box_alignment=(1,1))
plt.gca().add_artist(ab)

plt.savefig(ffolder+'/'+identifier+'_'+'SPL_r'+ DroneID  + segmented_based+'_ev_'+str(event)+'.jpg', format='jpg',dpi=1200)
#########
#%%% END
print(band)
print('AMplitude [dB]')
print("{:.1f} {:.1f}".format(plt_Level_min, plt_Level_max))
print('tetha')
print("{:.1f} {:.1f}".format(deg_th[2],deg_th[-3]))
print('phi')
print("{:.1f} {:.1f}".format(deg_ph[0],deg_ph[-1]))

#%%Dictionary for configuration figures
######################
BP_SPL={"identifier" :identifier,
          "1_3_ob_BP"  :L_by_band_chunk_BP,
          "1_3_bands"  :freq,
          "OASPL_BP"   :L_by_OASPL_chunk_BP,
          "theta"      :deg_th[2:-2],
          "phi"        :deg_ph}
import pickle #https://wiki.python.org/moin/UsingPickle
#save dictionary to pickle file
pickle.dump( BP_SPL, open(ffolder+'/'+identifier+"_BP_SPL.p", "wb" ) )