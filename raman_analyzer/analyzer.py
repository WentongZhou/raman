import lmfit
import os
import numpy as np
from matplotlib import pyplot as plt
import rampy as rp
import scipy
from scipy import signal
import warnings
warnings.filterwarnings("ignore")
class raman_analyzer:
    def __init__(self,name,min,max,filter,noise_min=1700,noise_max=2400):
        self.name = name
        self.min = min
        self.max = max
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.filter = filter
        self.peak_pos = []
        if self.name[-2] == 'p':
            self.spectrum = np.genfromtxt(self.name,delimiter=',')
        else:
            self.spectrum = np.genfromtxt(self.name)
        self.spectrum_resample =[]
        self.spectrum_corr = []
        self.spectrum_fit = []
        self.ese0 = 0.0
        self.sigma = 0.0
        self.peak_threshold = 0.0
        self.y_base = []
        self.peak_wavenumber = []
        self.peak_signal = []
        self.raman()
        self.peak_finder()
        self.peak_filtered = []
        self.peak_pos_filtered = []
        self.signal_filtered = []
        self.peak_organizer()
        self.fwhm = []
        self.fwhm_hline =[]
        self.peak_fwhm()
        self.noise_filter()

    def raman(self):
        x_new = np.arange(self.min, self.max, 0.5)
        y_new = rp.resample(self.spectrum[:,0], self.spectrum[:,1], x_new)
        self.spectrum_resample = np.vstack((x_new,y_new)).T
        y_smo_10 = rp.smooth(self.spectrum_resample[:,0],self.spectrum_resample[:,1],method="whittaker",Lambda=3000,window_length=7)
        self.spectrum_resample = np.vstack((x_new,y_smo_10)).T
        d = scipy.signal.argrelextrema(self.spectrum_resample, np.less)
        points = len(d[0])
        bir=np.zeros((points,2))
        for i in range(len(d[0])):
            wavenumber=self.spectrum_resample[d[0][i]][0]
            bir[i][0]=wavenumber
            bir[i][1]=wavenumber+5
        y_corr, self.y_base = rp.baseline(self.spectrum_resample[:,0],self.spectrum_resample[:,1],bir,'drPLS')
        x = self.spectrum_resample[:,0]
        x_fit = x[np.where((x > self.min)&(x < self.max))]
        y_fit = y_corr[np.where((x > self.min)&(x < self.max))]
        self.spectrum_corr = np.column_stack((x_fit,y_fit))
        self.ese0 = np.sqrt(abs(y_fit[:,0]))/abs(y_fit[:,0]) # the relative errors after baseline subtraction
        y_fit[:,0] = y_fit[:,0]/np.amax(y_fit[:,0])*10 # normalise spectra to maximum intensity, easier to handle
        self.spectrum_fit = np.column_stack((x_fit,y_fit))
        self.sigma = abs(self.ese0*y_fit[:,0])
    def peak_finder(self):
        peaks_1 = scipy.signal.find_peaks(self.spectrum_fit.T[1])
        saddles = scipy.signal.argrelmin(self.spectrum_fit.T[1])
        peaks = np.insert(peaks_1[0],0,saddles[0])
        peaks = np.sort(peaks)
        self.peak_pos = peaks
        for peak in peaks:
            self.peak_wavenumber.append(self.spectrum_fit.T[0][peak])
            self.peak_signal.append(self.spectrum_fit.T[1][peak])
    def peak_organizer(self):
        for i in range(len(self.peak_wavenumber)):
            if self.peak_signal[i] > self.filter:
                self.peak_pos_filtered.append(self.peak_pos[i])
                self.peak_filtered.append(self.peak_wavenumber[i])
                self.signal_filtered.append(self.peak_signal[i])
    def peak_fwhm(self):
        pw = signal.peak_widths(self.spectrum_fit[:,1],self.peak_pos_filtered,rel_height=0.495)
        pw_1 = signal.peak_widths(self.spectrum_fit[:,1],self.peak_pos_filtered,rel_height=0.97)
        f = scipy.interpolate.interp1d(range(len(self.spectrum_fit[:,0])),self.spectrum_fit[:,0])
        x_left = f(pw[2])
        x_right = f(pw[3])
        self.fwhm = x_right - x_left
        self.fwhm_hline = np.vstack((pw[1],x_left,x_right))
    def noise_filter(self):
        noise = self.spectrum_fit[:, 1][np.where((self.spectrum_fit[:, 0] > self.noise_min) & (self.spectrum_fit[:, 0] < self.noise_max))]
        self.peak_threshold = (np.max(noise) - np.min(noise))
def raman_batch(g,min,max,filter,show=False,export=False,noise_min=1700,noise_max=2400):
    g = os.walk(g)
    peaks = []
    signals = []
    fwhm = []
    fwhm_test = []
    name = []
    peak_threshold = []
    for path,dir_list,file_list in g:
        file_list.sort(key=lambda x:int(x.split('.')[1]))
        for file_name in file_list:
            a = os.path.join(path,file_name)
            raman = raman_analyzer(a,min,max,filter,noise_min,noise_max)
            peaks.append(raman.peak_filtered)
            signals.append(raman.signal_filtered)
            peak_threshold.append(raman.peak_threshold)
            name.append(file_name)
            fwhm.append(raman.fwhm)
            fwhm_test.append(raman.fwhm_hline[0])
            x = raman.spectrum_fit[:,0]
            y = raman.spectrum_fit[:,1]
            spectrum_fit = np.column_stack((x,y))
            if export == True:
                spectrum_fit = np.savetxt('./'+file_name+'.txt',spectrum_fit,fmt='%10.5f')
            if show == True:
                raman_plot(raman,min,max)
    return peaks,signals,fwhm,fwhm_test,name,peak_threshold
def ratio_calculator(directory,seive,d,g,twod,noise):
    a_1,b_1,c_1,d_1,name_1,e_1 = raman_batch(directory,d[0],d[1],d[2],noise_min=d[0],noise_max=d[1])
    aaa = np.zeros((len(a_1),1))
    for i in range(len(a_1)):
        if len(a_1[i])>1:
            num = np.where(b_1[i]==np.max(b_1[i]))
            aaa[i] = a_1[i][num[0][0]]
        else:
            aaa[i] = a_1[i][0]
    a_2,b_2,c_2,d_2,name_2,e_2 = raman_batch(directory,g[0],g[1],g[2],noise_min=g[0],noise_max=g[1])
    a_3,b_3,c_3,d_3,name_3,e_3 = raman_batch(directory,twod[0],twod[1],twod[2],noise_min=twod[0],noise_max=twod[1])
    a_2 = np.array(a_2)
    a_3 = np.array(a_3)
    a = np.hstack((aaa,a_2,a_3))
    a_4,b_4,c_4,d_4,name_4,e_4 = raman_batch(directory,1200,3000,0.01,noise_min=noise[0],noise_max=noise[1])
    peaks = []
    signals = []
    for i in range(len(a_4)):
        for j in range(len(a_4[i])):
            for s in range(len(a[i])):
                if a_4[i][j] < a[i][s] + seive and a_4[i][j] > a[i][s] - seive:
                    peaks.append(a_4[i][j])
                    signals.append(b_4[i][j])
    names = np.array(name_4)
    names = np.hstack((names,'mean','std'))
    peaks = np.array(peaks)
    signals = np.array(signals)
    peaks = peaks.reshape(len(a),3)
    peaks_all = np.zeros((len(a)+2,3))
    signals = signals.reshape(len(a),3)
    signals_all = np.zeros((len(a)+2,3))
    twod_g = signals[:,2]/signals[:,1]
    d_g = signals[:,0]/signals[:,1]
    twod_g = np.hstack((twod_g,twod_g.mean(),twod_g.std()))
    d_g = np.hstack((d_g, d_g.mean(), d_g.std()))
    filter_all = []
    for num in range(len(peaks[:,0])):
        filter = e_4[num]/signals[:,1][num]
        filter_all.append(filter)
        if d_g[num] < filter:
            d_g[num] = 0
            peaks[:,0][num] = 0
    for i in range(len(peaks[0])):
        peaks_all[:,i] = np.hstack((peaks[:,i],np.delete(peaks[:,i],np.where(peaks[:,i]==0)).mean(),np.delete(peaks[:,i],np.where(peaks[:,i]==0)).std()))
    for i in range(len(signals[0])):
        signals_all[:,i] = np.hstack((signals[:,i],signals[:,i].mean(),signals[:,i].std()))
    c_3 = np.array(c_3)
    c_3 = np.vstack((c_3,c_3.mean(),c_3.std()))
    total = np.hstack((peaks_all,signals_all))
    total = np.column_stack((names,total,twod_g,d_g,c_3))
    np.savetxt(directory+'_peaks'+'.txt',total,fmt='%s')
    np.save(directory+'_peaks'+'.npy',total)
    return total, filter_all
def raman_plot(a,min,max):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(a.spectrum_resample[:,0],a.spectrum_resample[:,1],'ko',markersize=1,label='spectrum_resample')
    plt.plot(a.spectrum_corr[:,0],a.spectrum_corr[:,1],'ro',markersize=1,label='spectrum_corr')
    plt.plot(a.spectrum_resample[:,0],a.y_base,'g--',label='baseline',linewidth=2)
    plt.xlim(min,max)
    plt.xlabel("Raman shift, cm$^{-1}$", fontsize = 12)
    plt.ylabel("Normalized intensity, a. u.", fontsize = 12)
    plt.legend(fontsize=10)
    plt.subplot(1,2,2)  
    plt.plot(a.spectrum_fit[:,0],a.spectrum_fit[:,1],'#F5420a',marker='.',markersize=3,label='spectrum_fit')
    plt.scatter(a.peak_filtered,a.signal_filtered,s=100,c='#0af57c',marker='o')
    plt.hlines(*a.fwhm_hline,colors='#2596be').set_linewidth(2)
    plt.ylim(-1,11)
    plt.xlim(min,max)
    plt.title(a.name)
    plt.xlabel("Raman shift, cm$^{-1}$", fontsize = 12)
