# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 18:34:28 2026

@author: jacob
Creates a synthetic light curve of RR Lyr for 4.11.2025. The template is based on several brightness measurements between 5.6.2025 
and 18.7.2025. I downloaded the data from AAVSO: https://www.aavso.org/data-download
Specifications:
    Object: RR Lyr
    Start date: 2460827.5
    End date: 2460888.5
    Discrepant data incuded: yes
    Diff and step data included: yes
    Format: comma delimited

https://www.aavso.org/LCGv2/ can be used to view the data
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from hjd2utc import hjd2utc
from datetime import datetime
from astropy.time import Time
from copy import deepcopy
import helper

plt.close('all')
repo_root = helper.get_repo_root()

def load_aavso_V_data(filepath):
    df = pd.read_csv(filepath, sep=",", comment="#", na_values=["", " "])   #treats empty fields as NaN
    df = df[df["Band"] == "V"]
    return df

def get_JD_mag(df, disp=False):
    JD = np.array(df['JD'])
    mag = np.array(df['Magnitude'])
    if disp:
        plt.plot(JD, mag, '.g')
        plt.show()
    return JD, mag

def get_patches(JD, mag):
    JD_patches = []
    mag_patches = []
    patch_startInd = 0
    for i in range(1, len(JD)-1):
        if JD[i] - JD[i-1] > 1:
            JD_patches += [JD[patch_startInd : (i-1)]]
            mag_patches += [mag[patch_startInd : (i-1)]]
            patch_startInd = i
    return JD_patches, mag_patches

def running_avg(JD, mag, window_width):
    if window_width % 2 == 0:
        raise Exception('The window_width has to be uneven.')
    window = np.ones(window_width)
    mag_smooth = np.convolve(mag, window)
    mag_smooth = mag_smooth[(window_width-1) : -(window_width-1)] / window_width
    JD_smooth = JD[int(np.floor(window_width/2)) : -int(np.floor(window_width/2))]
    return JD_smooth, mag_smooth

def find_max_ind(mag):
    #set NaNs to a high (dark) magnitude value
    mag_modified = deepcopy(mag)
    mag_modified[np.isnan(mag_modified)] = 100
    
    #find index of maximum
    max_ind = int(np.where(mag_modified == np.min(mag_modified))[0][0])  #searches for the minimum, because at the brightness maximum, the magnitude is minimal
    return max_ind

def utc2jd(utc):
    t_utc = Time(utc)
    t_jd = t_utc.jd
    return t_jd

def jd2utc(jd):
    t_jd = Time(jd, format='jd', scale='utc')
    return t_jd




#%% load AAVSO data
filepath_010625_010825 = repo_root/'data/AAVSO all bands 1.6.2025 to 1.8.2025.txt'
df_010625_010825 = load_aavso_V_data(filepath_010625_010825)

JD_010625_010825, mag_010625_010825 = get_JD_mag(df_010625_010825, disp=False)


#%% separate into patches where data is available and shift them so that they perfectly overlap (shifts are eyeballed)
JD_050625 = JD_010625_010825[:96] + 35.11
mag_050625 = mag_010625_010825[:96]

JD_060625 = JD_010625_010825[96:323] + 34.0
mag_060625 = mag_010625_010825[96:323]

# JD_070625 = JD_010625_010825[323:463] + 32.875
# mag_070625 = mag_010625_010825[323:463]           #outlier

JD_080625 = JD_010625_010825[463:590] + 31.737
mag_080625 = mag_010625_010825[463:590] +.03        #changing the magnitude is not optimal, but the ovarall shape of the magnitude 
                                                    #variation is more important than the magnitude itself

# JD_090625 = JD_010625_010825[590:715] + 31
# mag_090625 = mag_010625_010825[590:715]           #outlier

JD_230625 = JD_010625_010825[715:1265] + 17.005
mag_230625 = mag_010625_010825[715:1265]

JD_090725 = JD_010625_010825[1265:1753] + 1.12
mag_090725 = mag_010625_010825[1265:1753]

JD_100725 = JD_010625_010825[1753:2204] + 0
mag_100725 = mag_010625_010825[1753:2204]

JD_180725 = JD_010625_010825[2204:] - 7.94
mag_180725 = mag_010625_010825[2204:]


# createa list of patches for JD and mag. patch_date will be used for the legend of the first overview plot
JD_patches = [JD_050625, JD_060625, JD_080625, JD_230625, JD_090725, JD_100725, JD_180725]
mag_patches = [mag_050625, mag_060625, mag_080625, mag_230625, mag_090725, mag_100725, mag_180725]
for m in range(len(JD_patches)):
    JD_patches[m], mag_patches[m] = running_avg(JD_patches[m], mag_patches[m], 21)
patch_date = ['5.6.2025', '6.6.2025', '8.6.2025', '23.6.2025', '9.7.2025', '10.7.2025', '18.7.2025']



# first overview plot for the patches; used for eyeballing the shifts
# fig_patches = plt.figure()
# for m in range(len(JD_patches)):
#     plt.plot(JD_patches[m], mag_patches[m], '-', label=patch_date[m])
# plt.gca().invert_yaxis()
# plt.legend()
# plt.show()



#%% create a single time axis and calculate the mean magnitude at each time step
t_step = np.mean(np.diff(JD_180725))
t_min = np.min(JD_080625)
t_max = np.max(JD_050625)
t_axis_JD = np.arange(t_min, t_max, t_step)

JD = np.array([x for patch in JD_patches for x in patch])
mag = np.array([x for patch in mag_patches for x in patch])

mean_mag = []
smoothing_factor = 4
for i in range(len(t_axis_JD)):
    range_min = t_axis_JD[i] - smoothing_factor*t_step
    range_max = t_axis_JD[i] + smoothing_factor*t_step
    inds = np.where((range_min <= JD) & (JD < range_max))[0]    #this binning is necessary, because JDs of different patches may not have exactly the same values
    mean_mag.append(np.mean(mag[inds]))
mean_mag = np.array(mean_mag)
t_axis = t_axis_JD - t_min      #drop the dependence on a reference date


# second overview plot: mean magnitude
# fig_mean_mag = plt.figure()
# plt.plot(t_axis, mean_mag, '-r')
# plt.plot(t_axis+.47, mean_mag, '-b')   #second light curve for eyeballing the length of the missing part
# plt.gca().invert_yaxis()
# plt.title('Mean magnitude')
# plt.show()


#extend with NaNs to reach a full oscillation period. The NaNs resemble the missing part of the light curve
t_axis_new = np.arange(0, .47, t_step)    #the .47 are eyeballed using the second overview plot
extension = len(t_axis_new) - len(mean_mag)
mean_mag_new = np.append(mean_mag, np.full(extension, np.nan))


#reshape the light curve so that the first value is the maximum. This makes the chaining easier (see below)
max_ind = find_max_ind(mean_mag_new)
mean_mag_new = np.roll(mean_mag_new, -max_ind)  #takes everything after max_ind (including max_ind) and puts it to the beginning of the array





#%% scale the light curve to match the true average period 
# (freq: f_0 from https://arxiv.org/pdf/1011.5908, page 7)

freq = 1.76416    #oscillations per day
period = 1 / freq #in days
t_axis_new = t_axis_new / np.max(t_axis_new) * period


#%% chain several oscillations together starting with the nearest known brightness maximum
# (from https://rr-lyr.irap.omp.eu/dbrr/rrdb-V2.0_08.3.php?RR+Lyr& → this website lists the dates of known brightness maxima)

# find the brightness maximum that is closest to 4.11.2025
_ = hjd2utc(hjd=2460979.4531, ra_deg=291.36629, dec_deg=42.78436, print_result=True)  #→ 30.10.2025, 22:52:30.075
known_max_1 = datetime.strptime('30.10.2025, 22:52:30.075', '%d.%m.%Y, %H:%M:%S.%f')
_ = hjd2utc(hjd=2460995.3201, ra_deg=291.36629, dec_deg=42.78436, print_result=True)  #→ 15.11.2025, 19:41:59.154
known_max_2 = datetime.strptime('15.11.2025, 19:41:59.154', '%d.%m.%Y, %H:%M:%S.%f')

timediff = known_max_2 - known_max_1
n_periods = timediff.total_seconds() / 86400 / period   # = 27.993 → almost an exact multiple of the period → great!


# third overview plot: two oscillations together
# fig_two_oscis = plt.figure()
# plt.plot(np.hstack((t_axis_new, t_axis_new+period)), np.hstack((mean_mag_new, mean_mag_new)), 'r')
# plt.gca().invert_yaxis()
# plt.show()


# chain oscillations between known_max_1 and known_max_2
n_periods = round(n_periods)
t_axis_stack = []
mean_mag_stack = []
for n in range(n_periods):
    t_axis_stack = np.append(t_axis_stack, t_axis_new + period*n)
    mean_mag_stack = np.append(mean_mag_stack, mean_mag_new)
t_axis_stack += utc2jd(known_max_1)   #convert back to julian date



# fourth overview plot: chained oscillations
# fig_stack = plt.figure(figsize=(20,8))
# plt.plot(t_axis_stack, mean_mag_stack, 'g')
# plt.gca().invert_yaxis()
# plt.show()



#%% cut out the light curve for 4.11.2025 and create a beautiful plot
start_jd = 2460983.5   #UTC 2025-11-04T00:00:00 = JD 2460983.5
start_utc = datetime(2025, 11, 4, 0, 0, 0)
end_jd = 2460984.5     #UTC 2025-11-05T00:00:00 = JD 2460984.5
end_utc = datetime(2025, 11, 5, 0, 0, 0)

inds_041125 = np.where((start_jd <= t_axis_stack) & (t_axis_stack < end_jd))[0]
t_axis_JD_041125 = jd2utc(t_axis_stack[inds_041125]).iso
#t_axis_UTC_041125 = jd2utc(t_axis_JD_041125).to_datetime()
mean_mag_041125 = mean_mag_stack[inds_041125]

dti = pd.to_datetime(t_axis_JD_041125)
mag_df = pd.DataFrame(data={'t': dti, 'mag': mean_mag_041125})
mag_df.to_csv(repo_root/'data/rr_lyr_model.csv')

# fifth plot: light curve of 4.11.2025
fig_lightcurve = plt.figure(figsize=(12,8))
#plt.plot(t_axis_JD_041125, mean_mag_041125, 'r')
plt.plot(mag_df['t'], mag_df['mag'], 'r')
plt.gca().invert_yaxis()
locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)
plt.gca().xaxis.set_major_locator(locator)
plt.gca().xaxis.set_major_formatter(formatter)

plt.title('Synthetic Light Curve of RR Lyr of 4.11.2025 Filter: V')
plt.xlabel('time')
plt.ylabel('brightness / mag')
plt.savefig(repo_root/'figs/rr_lyr_ref_V_band_full.png')

plt.xlim([np.datetime64("2025-11-04 17:00"), np.datetime64("2025-11-04 22:00")])
plt.ylim([8.12, 7.7])
plt.savefig(repo_root/'figs/rr_lyr_ref_V_band_cut.png')

# ticks_jd = np.arange(start_jd, end_jd, 1/12)  #one tick every 2 hours
# tick_labels = jd2utc(ticks_jd).isot  #iso format 2025-11-04T02:00:00
# plt.xticks(ticks_jd, tick_labels, rotation=45)

#plt.show()












