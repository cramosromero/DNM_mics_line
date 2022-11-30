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

# %% Constants
# #############
n_mics  = 9
to_bands =[20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250,
        315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
        5000, 6300, 8000, 10000, 12500, 16000, 20000]
# %%  Noise RECORDINGS .mat files
# ################################
time_slice = 2.5
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
im_drone_file_over = os.getcwd() +"\\UASimg\\"+ DroneID+PYL+'_over.png'#"drone_Q_over.png

print(identifier)
fly_speed = float(OPE[-2:])

# ACCCES DATA_ACU SINGLE EVENT
event   = 0 # {1,2,3,4,5,6,7,8,9... nEve}
acu_metric = 'LAFp' # LAFp, LZfp

Fs, TT, DATA_raw, DATA_acu = FileTools.data_single_events (files[event-1], n_mics, acu_metric)

# %%%% PLOTS all microphones, single events
# ################################
"""PLOTS all microphones, single events"""
fig, (ax0) = plt.subplots()

for mics in range(DATA_raw.shape[1]):
    plt.plot(TT, DATA_raw[:,mics],**next(lc),linewidth=1,label='Ch {}'.format(mics+1))

plt.title(DroneID + ' Event {}'.format(event))
plt.legend(loc='upper right', borderpad=0.1,frameon=True)
ax0.get_legend().set_title("Microphone")
plt.ylabel(' Amplitude [Pa]')
plt.xlabel('Time [s]')
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
fig, (ax0) = plt.subplots()

for eve in range(DATA_mic.shape[1]):
    plt.plot(TT, DATA_mic[:,eve],**next(lc),linewidth=1,label='{}'.format(eve+1))
plt.title('Microphone {}'.format(microphone))    
plt.legend(loc='upper right', borderpad=0.1,frameon=True)
ax0.get_legend().set_title(DroneID +" Events")
plt.ylabel(' Amplitude Pa')
plt.xlabel('Time [s]')
# plt.show()

# %% SIGNAL SEGMENTATION 
##############################
""" SIGNAL SEGMENTATION based on LA_max value of the medianfilter signal
    This segment could be analized with FFT to obtain the PSD, then the band content"""
    
segmented_based = 'time' #{'time','level'}   

""" Based on time slice with max value centered"""
# time_slice = 5.8

TIME_r, vec_time, Data_raw_segmented, Data_acu_segmented = TimeTools.segment_time_ADJUST (time_slice, Fs,
                                                                                          DATA_raw_events, DATA_acu_events)
# %%%% PLOT segmented by time
"""PLOTS DATA_mic same microphone, all events"""
fig, (ax0) = plt.subplots()
for eve in range(Data_raw_segmented.shape[0]):
    plt.plot(TIME_r, Data_acu_segmented[eve,:,microphone-1],**next(lc),linewidth=1,label='{}'.format(eve))
plt.title(DroneID + ' Microphone {}'.format(microphone))    
plt.legend(loc='upper right', borderpad=0.1,frameon=True)
ax0.get_legend().set_title("Events")
plt.ylabel(' Amplitude [Pa]')
plt.xlabel('Time [s]')
plt.show()
    
# %%%% If data selectio depends on threshold 
if segmented_based == 'time':
    pass

elif segmented_based == 'level': 
    """ Based on time signals lower than the Lmax"""
    THRESH = 20
    TIME_r, vec_time, Data_raw_segmented_trh, Data_acu_segmented_trh  = TimeTools.segment_THRESH(THRESH,Fs,Data_raw_segmented, Data_acu_segmented)
       
    # PLOT segmented by threshold
    """PLOTS DATA_mic same microphone, all events"""
    # fig, (ax0) = plt.subplots()
    # for eve in range(Data_raw_segmented.shape[0]):
    #     plt.plot(Data_acu_segmented_trh[eve,:,microphone-1],**next(lc),linewidth=1,label='{}'.format(eve))
    # plt.title(DroneID + ' Microphone {}'.format(microphone))    
    # plt.legend(loc='upper right', borderpad=0.1,frameon=True)
    # ax0.get_legend().set_title("Events")
    # plt.ylabel(' Amplitude [Pa]')
    # plt.xlabel('Time [s]')
    # plt.show()
    
    Data_raw_segmented = Data_raw_segmented_trh
    Data_acu_segmented = Data_acu_segmented_trh


# %% FLYby - TIME WINDOWING
################################
""" FLYby - TIME WINDOWING"""

s0 = Data_raw_segmented.shape[1]//2    #   central sample into the complete time_slice
time_slice =  round(Data_raw_segmented.shape[1]/Fs)
""" Constants for FFT-PSD """
Ndft    = 2**13 #{recomended 2**13}
df      = Fs/Ndft
p_ref   = 20e-6
WINDOW = 'hann'

Nw = time_slice*Fs//Ndft 

if (Nw % 2) == 0: Nn=Nw-1 
ovrl = 0                                # overlap in %
sw = Data_raw_segmented.shape[1]//Nw    # samples on time window
tw = sw/Fs                              # window's time

DX = np.zeros(int(Nw))
PHI = np.zeros(int(Nw))

LEVELS_th_ph = []
STD_th_ph = []

fig, (ax0) = plt.subplots()

for win_ord in range(Nw):
    # win_ord -> {0:Nw-1}
    
    si = int(win_ord * sw)              #   initial sample
    sf = int((win_ord+1) * sw)          #   final sample   
    sc = si + (sf-si)//2                #   central sample
    
    
    DATA_raw_wint_segmented = Data_raw_segmented[:,si:sf,:]
    DATA_acu_wint_segmented = Data_acu_segmented[:,si:sf,:]
    vec_time_wint_segmented = vec_time[si:sf]
    
    dx = fly_speed*(np.linalg.norm(s0-sc)/Fs)
    
    plt.plot(vec_time_wint_segmented,DATA_acu_wint_segmented[-1,:,4])
    #save this figure manually 
    #%%%% CALC: 1/3 octave band
    #####################
    fraction = 3
    DATA_SPLoverall, DATA_SPLs, freq = FreqTools.calc_SPLs(DATA_raw_wint_segmented,Fs, fraction, limits=[20, 20000], show=0)
    
    #%%%%%% BAND Selection for Directivity 
    band = 'overall'
    
    
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
        #%%%%%% PLOTS 1/3 octave band 
        #############
        """ PLOTS """
        # x_freq = np.arange(len(freq))  # the label locations
        # labels = to_bands
        # fig, (ax0) = plt.subplots()
        # m1 = 1
        # m2 = 5
        # m3 = 7
        # ax0.bar(x_freq, spl_s_mean[:,m1-1], align='center', color='C1', alpha=1, label='mic {}'.format(m1))
        # ax0.bar(x_freq, spl_s_mean[:,m2-1], align='center', color='C2', alpha=0.5, label='mic {}'.format(m2))
        # ax0.bar(x_freq, spl_s_mean[:,m3-1], align='center', color='C3', alpha=0.5, label='mic {}'.format(m3))
        
        # plt.errorbar(x_freq, spl_s_mean[:,m1-1],yerr=2*spl_s_std[:,m1-1],color='C1',alpha=1, fmt='none', mfc='o')
        # plt.errorbar(x_freq, spl_s_mean[:,m2-1],yerr=2*spl_s_std[:,m2-1],color='C2',alpha=1, fmt='none', mfc='o')
        # plt.errorbar(x_freq, spl_s_mean[:,m3-1],yerr=2*spl_s_std[:,m3-1],color='C3',alpha=1, fmt='none', mfc='o')
        
        # ax0.set_xticks(x_freq, labels)
        # ax0.set_ylabel('$L_p[dB]$ re.$(20 \mu Pa)^2$')
        # ax0.set_xlabel('Frequency [Hz]')
        # ax0.set_title(DroneID + ' 1/{} Octave Band'.format(fraction))
        
        # ax0.set_ylim([20,70])
        # ax0.yaxis.grid(which='minor', color='#EEEEEE', linestyle='--', linewidth=0.8)
        # plt.minorticks_on()
        # plt.xticks(rotation=90)
        # plt.tight_layout()
        # plt.legend(loc='upper right', borderpad=0.1,frameon=True)
        # plt.show()
    #%%%%%% Overall level for Directivity 
        
    elif (type(band) == str and band =='overall') == True: ## to calculate the overall SPL from bands
        b_index = to_bands.index(1000)
        band = 1000 #for atmospheric correction
        
        spl_s_mean = np.mean(DATA_SPLoverall[:,:,:],axis=0)
        spl_s_std = np.std(DATA_SPLoverall[:,:,:],axis=0)
        
        SPL_event_mic = DATA_SPLoverall[:,:,:] # data [events]
        label = '$SPL[dB]$ Overall'
    #%%%% DEPROPAGATION
    """ SPL DEPROPAGATION"""
    rad_to_deprop = 1 #[m]
    heightAGL = HAGL #[m]
    
    Lmax_array = SPL_event_mic
    
    L_array_depropagated, thetas, dist_ground_mics, phi_dx = AircraftTools.depropagator(Lmax_array, heightAGL, rad_to_deprop, band, dx)
    
    L_array_deprop_mean = np.squeeze(np.mean(L_array_depropagated, axis=0))
    L_array_deprop_std = np.squeeze(np.std(L_array_depropagated, axis=0))
    
    """ DI Marshal"""
    DI_events = AircraftTools.directivity(L_array_depropagated)
    
    DI_mean = np.squeeze(np.mean(DI_events, axis=0))
    DI_std = np.squeeze(np.std(DI_events, axis=0))
    
    """Levels After Depropagation"""
    LEVEL_TO_PLOT = L_array_deprop_mean
    STD_TO_PLOT = L_array_deprop_std
        
    #%%%% PLOTS DIRECTIVITY-LEVEL mean std, all events - all mics
    #########################
    """PLOTS DIRECTIVITY-LEVEL mean std, all events - all mics"""
    
    # dfor = next(lc) #draw_format
    # dfor = dfor["color"] + dfor["linestyle"]
    
    # from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    # from matplotlib.cbook import get_sample_data
    # from matplotlib.image import imread
    
    # theta_ref = np.max(np.array(thetas))
    # theta_ref_radians = (theta_ref/180)*np.pi
    
    # theta = np.linspace((np.pi/2)-theta_ref_radians, (np.pi/2)+theta_ref_radians, len(thetas))
    
    # fig = plt.figure(figsize=(4,4))
    # ax = fig.add_subplot(111, polar=True)
    # ax.plot(theta, LEVEL_TO_PLOT, dfor, label='$SPL[dB]:{} Hz$'.format(band))
    # ax.legend(loc='lower center')
    
    # plt.fill_between(theta, LEVEL_TO_PLOT - L_array_deprop_std,
    #                   LEVEL_TO_PLOT + L_array_deprop_std,
    #                   color='gray', alpha=0.2)
    
    # r_min = TimeTools.round_up(np.min(LEVEL_TO_PLOT),10) - 10
    # r_max = TimeTools.round_up(np.max(LEVEL_TO_PLOT),10)
    # r_r = r_min-5
    
    # ax.set_rmin(r_min)
    # ax.set_rmax(r_max)
    # ax.set_rorigin(r_r)
    
    # ax.set_rgrids(np.linspace(r_min,r_max,int((r_max-r_min)/5+1)))
    
    # ax.set_thetamin(30)
    # ax.set_thetamax(150)
    # thetas_t = [str(angle) + '\N{DEGREE SIGN}' for angle in thetas] #thetas labels
    # ax.set_xticks(((np.arange(len(thetas_t))*15)+30)*np.pi/180)
    
    # ax.set_xticklabels(thetas_t)
    
    # ax.set_title(DroneID + ' Directivity pattern'+'\n'+
    #              identifier +'\n'+ acu_metric +
    #              ' $R=' + str(rad_to_deprop)+'[m]$')
    
    # """ Add drone imagen """
    # im = imread(get_sample_data(im_drone_file, asfileobj=False))
    # oi = OffsetImage(im, zoom=0.15)
    # ab = AnnotationBbox(oi, xy=(np.pi*90/180, r_min), frameon=False, clip_on=False, xybox=(90,r_r))#, box_alignment=(1,1))
    # plt.gca().add_artist(ab)
    
    # ax.set_theta_offset(np.pi) #invert the plot in vertical axis
    
    # ax.yaxis.grid(which='minor', color='#ECECEC', linestyle='--', linewidth=0.8)
    
    # plt.minorticks_on()
    # plt.tight_layout()
    # plt.show()
    
    #%%%% PLOTS DIRECTIVITY-INDEX mean std, all events - all mics
    #############################
    """PLOTS DIRECTIVITY-INDEX mean std, all events - all mics"""
    
    # fig = plt.figure(figsize=(4,4))
    
    # ax = fig.add_subplot(111, polar=True)
    # ax.plot(theta, DI_mean, dfor, label='$DI[dB]:{} [Hz]$'.format(band))
    # ax.legend(loc='lower center')
    
    # plt.fill_between(theta, DI_mean - DI_std,
    #                   DI_mean + DI_std,
    #                   color='gray', alpha=0.2)
    
    # r_min = TimeTools.round_up(np.min(DI_mean),3) - 6
    # r_max = TimeTools.round_up(np.max(DI_mean),3)
    # r_r = r_min-3
    
    # ax.set_rmin(r_min)
    # ax.set_rmax(r_max)
    # ax.set_rorigin(r_r)
    
    # ax.set_rgrids(np.linspace(r_min,r_max,int((r_max-r_min)/3+1)))
    
    # ax.set_thetamin(30)
    # ax.set_thetamax(150)
    # thetas_t = [str(angle) + '\N{DEGREE SIGN}' for angle in thetas] #thetas labels
    # ax.set_xticks(((np.arange(len(thetas_t))*15)+30)*np.pi/180)
    
    # ax.set_xticklabels(thetas_t)
    
    # ax.set_title(DroneID + 'Directivity Index'+'\n'+ identifier +'\n'+
    #               '$R={}[m]$'.format(rad_to_deprop), multialignment='center')
    # # ax.minorticks_on()
    # """ Add drone imagen """
    # im = imread(get_sample_data(im_drone_file, asfileobj=False))
    # oi = OffsetImage(im, zoom=0.15)
    # ab = AnnotationBbox(oi, xy=(np.pi*90/180, r_min),frameon=False, clip_on=False,xybox=(90,r_r))#, box_alignment=(1,1))
    # plt.gca().add_artist(ab)
    
    # ax.set_theta_offset(np.pi) #invert the plot in vertical axis
    
    # ax.yaxis.grid(which='minor', color='#ECECEC', linestyle='--', linewidth=0.8)
    
    # plt.minorticks_on()
    # plt.tight_layout()
    # plt.show()

    
    #####################    
    #%% COMPLETE PATH SOUND LEVELES and STD
    DX[win_ord] = dx        # distances to mic-line
    PHI[win_ord] = phi_dx   # polar angles
    
    LEVELS_th_ph.append(LEVEL_TO_PLOT) # Levels
    STD_th_ph.append(STD_TO_PLOT) # STD
    
LEVELS_th_ph = np.array(LEVELS_th_ph)
STD_th_ph = np.array(STD_th_ph)

if Count =='uw' : ##flip the aray if the drone flyes up-wind (for the order in the Sphere)
    LEVELS_th_ph = np.flip(LEVELS_th_ph, 1)
    STD_th_ph = np.flip(STD_th_ph, 1)
    
# limits for colorplots  
# plt_Level_min = 55 # background noise
plt_Level_min = np.min(LEVELS_th_ph)
plt_Level_max = np.max(LEVELS_th_ph)
# plt_Level_max = np.max(LEVELS_th_ph)//10*10+6#85 # maximum value of SPL

#%% PLOTS SEMIESPHERES - (complete Path)
#########################
"""PLOTS SEMIESPHERES, all events - all mics"""
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
# %%%% Coordinates
## Azimut angle 
th_min = 0
th_max = 180
## Polar angle
ph_min = -90
ph_max = 90

j_15 = 1 + n_mics+ (180-(n_mics*15))//15 # Mics angle, jump+ 15 deg.
th   = np.linspace(th_min*np.pi/180, th_max*np.pi/180, j_15 ) 

all_phi_rad = 90-(PHI*180/np.pi) # Path angle. coordinathes acording with NORAH
ph = np.concatenate([all_phi_rad[0:all_phi_rad.shape[0]//2]*-1, all_phi_rad[all_phi_rad.shape[0]//2:]]) 
ph = ph*np.pi/180

th, ph = np.meshgrid(th,ph)

## NORAH coordinates
x = 10 * np.cos(th)
y = -10 * np.sin(th) * np.sin(ph)
z = HAGL - (10 *np.sin(th) * np.cos(ph))

## BIG semihespheres 
#  DL Directivity levels
#  DSTD Distectivity STD
DL_theta_phi = np.ones(th.shape)*(plt_Level_min-6) # 30 db as background noise
DSTD_theta_phi = np.ones(th.shape)*(-1) # 
# %%%% LEVELS LOCATION on the SEMIESPHERE
for i_phi in range(LEVELS_th_ph.shape[0]):
    DL_theta_phi[i_phi,2:11] = LEVELS_th_ph[i_phi,:]
    DSTD_theta_phi[i_phi,2:11] = STD_th_ph[i_phi,:]

# Level_min = np.min(DL_theta_phi)
# Level_max = np.max(DL_theta_phi)

mask = np.ones(th.shape) #zones without information
masked = np.ma.masked_where(DL_theta_phi>plt_Level_min-1, mask)#zones without information
# %%%% PLOT for directivity 3D
# %%%%%% Colormap
cmap = plt.colormaps["jet_r"].reversed() 
strength = DL_theta_phi

# norm = colors.Normalize(vmin = plt_Level_min, vmax = plt_Level_max, clip = False)
norm = colors.Normalize(vmin = np.min(strength), vmax = np.max(strength), clip = False)

# %%%%%% Figure
fig = plt.figure(figsize=(8,8))
plt.figaspect(1)
ax = fig.add_subplot(111, projection='3d')
""" axis directivity"""
ec = ax.plot_surface(x, y, z, edgecolor='none',facecolors = cmap(norm(strength)),
                      cmap=cmap, antialiased=False)

# """ axis reference"""
# d_ref = 2
# l_1_x = [0,0] #Z
# l_1_y = [0,0]
# l_1_z = [HAGL,HAGL-d_ref]

# l_2_x = [0,d_ref] #X
# l_2_y = [0,0]
# l_2_z = [HAGL,HAGL]

# l_3_x = [0,0] #Y
# l_3_y = [0,d_ref]
# l_3_z = [HAGL,HAGL]

# ax.plot(l_1_x, l_1_y, l_1_z, color='#25A4DF')   # extend in z direction
# ax.plot(l_2_x, l_2_y, l_2_z, color='#8D5012')   # extend in x direction microphones
# ax.plot(l_3_x, l_3_y, l_3_z,'k')   # extend in y direction


ax.set_box_aspect([1,1,1])

# %%%%%% Mic-line as a reference
ax.plot([dist_ground_mics[0]*10 / dist_ground_mics[0]],[0],[-10],marker='.', color='#6E2C00', alpha=0.9)

for nm in range(1, len(dist_ground_mics), 1):
    xcor = dist_ground_mics[nm]*10 / dist_ground_mics[0]
    if nm > len(dist_ground_mics)//2:
        xcor= xcor*-1
    ax.plot([xcor],[0],[-10],marker='.', color='#B2783D', alpha=0.5)
    
    
# %%%%%% Figure axis
ax.set_axis_off()
plt.title('SPL Hemispheres'+'\n'+ DroneID+'\n'+label)
ax.set_xlabel('N-S X axis')
ax.set_ylabel('W-E Y axis')
ax.set_zlabel('Z axis')

ax.view_init(30,-230)
plt.tight_layout()

plt.savefig(ffolder+'/'+identifier+'_'+'SPLHemi'+ DroneID + label+'.jpg', format='jpg',dpi=1200)
plt.show()

# %%%%%% Colorbar
"""COLORBAR"""
import matplotlib.pyplot as plt
import matplotlib as mpl

fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)

cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal',
             label ='[dB]'.format(band), alpha=0.8)
cbt = cbar.get_ticks().tolist()
cbt[0]='$no$ $data$'
cbar.ax.set_xticklabels(cbt)

plt.tight_layout()
plt.savefig(ffolder+'/'+identifier+'_'+'SPLHemi'+ DroneID + label+'cb.jpg', format='jpg',dpi=1200)
# For informative


# %%%%%% "unwrapped"-directivity and STD
from matplotlib.ticker import AutoMinorLocator
th_ticks_label = ['','','1','2','3','4','5','6','7','8','9','','']
deg_th = np.array([-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90])
if Count  == 'uw':
    th_ticks_label = ['','','9','8','7','6','5','4','3','4','1','','']
    # deg_th = np.flip(deg_th)

deg_ph = -1*ph[:,0]*180/np.pi

fig, ax = plt.subplots(figsize=(3.7,8))
plt.grid(False)

plt.imshow(DL_theta_phi, cmap=cmap,interpolation ='gaussian')#, interpolation ='none', alpha=0.8)
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
plt.title('DI '+label)

plt.imshow(masked, 'gray', interpolation='none', alpha=0.8) #zones without information

secax = ax.secondary_xaxis('top') #second axis THETA
secax.set_xticks(np.arange(0, len(th_ticks_label), 1),deg_th, rotation=45,fontsize=8)
secax.set_xlabel('$\Theta$')
plt.tight_layout()
""" Add drone imagen """
im = imread(get_sample_data(im_drone_file_over, asfileobj=False))
oi = OffsetImage(im, zoom=0.3)
ab = AnnotationBbox(oi, xy=(len(th_ticks_label)//2,deg_ph.shape[0]//2-0.5), frameon=False, clip_on=False)#, box_alignment=(1,1))
plt.gca().add_artist(ab)

plt.savefig(ffolder+'/'+identifier+'_'+'SPL_r'+ DroneID + label+'.jpg', format='jpg',dpi=1200)
#########
#########
fig, ax = plt.subplots(figsize=(3.7,8))
plt.grid(False)

cax = plt.imshow(DSTD_theta_phi, cmap=cmap,interpolation ='gaussian')#, interpolation ='none', alpha=0.8)
cbar = plt.colorbar(orientation='horizontal', alpha=0.8,label ='$std_{SPL}$ [dB]',
                    fraction=0.075, pad=0.07, extend='max')#,ticks=[0,1,2,3,4,5,6])#, location='bottom')
plt.clim(0, 6);
# cbt = cbar.get_ticks().tolist()
# cbt[0]='$no$ $data$'
# cbar.ax.set_xticklabels(cbt)  # horizontal colorbar
# plt.gca().invert_yaxis()
# plt.xticks()
plt.ylabel(r'$\phi$')
plt.yticks(np.arange(0, deg_ph.shape[0], 1),np.round(deg_ph,1),fontsize=8)
plt.xlabel('Microphone')#r'$\Theta$')
plt.xticks(np.arange(0, len(th_ticks_label), 1),th_ticks_label, rotation=0,fontsize=8)
plt.title('$std_{Dir}$ '+label)

plt.imshow(masked, 'gray', interpolation='none', alpha=0.8) #zones without information

secax = ax.secondary_xaxis('top') #second axis THETA
secax.set_xticks(np.arange(0, len(th_ticks_label), 1),deg_th, rotation=45,fontsize=8)
secax.set_xlabel('$\theta$')
plt.tight_layout()
""" Add drone imagen """
im = imread(get_sample_data(im_drone_file_over, asfileobj=False))
oi = OffsetImage(im, zoom=0.3)
ab = AnnotationBbox(oi, xy=(len(th_ticks_label)//2,deg_ph.shape[0]//2-0.5), frameon=False, clip_on=False)#, box_alignment=(1,1))
plt.gca().add_artist(ab)

plt.savefig(ffolder+'/'+identifier+'_'+'SPLstd_r'+ DroneID + label+'.jpg', format='jpg',dpi=1200)

#%%% END
print(band)
print('AMplitude [dB]')
print("{:.1f} {:.1f}".format(plt_Level_min, plt_Level_max))
print('tetha')
print("{:.1f} {:.1f}".format(deg_th[2],deg_th[-3]))
print('phi')
print("{:.1f} {:.1f}".format(deg_ph[0],deg_ph[-1]))
