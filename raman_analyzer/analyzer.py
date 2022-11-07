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
    def __init__(self,name,min,max,filter):
        self.name = name
        self.min = min
        self.max = max
        self.filter = filter
        self.peak_pos = []
        self.spectrum = np.genfromtxt(self.name)
        self.spectrum_resample =[]
        self.spectrum_corr = []
        self.spectrum_fit = []
        self.ese0 = 0.0
        self.sigma = 0.0
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
def raman_batch(g,min,max,filter):
    g = os.walk(g)
    peaks = []
    signals = []
    fwhm = []
    fwhm_test = []
    name = []
    for path,dir_list,file_list in g:
        file_list.sort(key=lambda x:int(x.split('.')[1]))
        for file_name in file_list:
            a = os.path.join(path,file_name)
            raman = raman_analyzer(a,min,max,filter)
            peaks.append(raman.peak_filtered)
            signals.append(raman.signal_filtered)
            name.append(file_name)
            fwhm.append(raman.fwhm)
            fwhm_test.append(raman.fwhm_hline[0])
            x = raman.spectrum_fit[:,0]
            y = raman.spectrum_fit[:,1]
            spectrum_fit = np.column_stack((x,y))
            spectrum_fit = np.savetxt('./'+file_name+'.txt',spectrum_fit,fmt='%10.5f')
            # raman_plot(raman,min,max)
    return peaks, signals,fwhm,fwhm_test,name
def ratio_calculator(directory,filter):
    a_1,b_1,c_1,d_1,name_1 = raman_batch(directory,1200,1400,7)
    aaa = np.zeros((len(a_1),1))
    for i in range(len(a_1)):
        if len(a_1[i])>1:
            num = np.where(b_1[i]==np.max(b_1[i]))
            aaa[i] = a_1[i][num[0][0]]
        else:
            aaa[i] = a_1[i][0]
    a_2,b_2,c_2,d_2,name_2 = raman_batch(directory,1500,1700,7)
    a_3,b_3,c_3,d_3,name_3 = raman_batch(directory,2600,2900,7)
    a_2 = np.array(a_2)
    a_3 = np.array(a_3)
    a = np.hstack((aaa,a_2,a_3))
    a_4,b_4,c_4,d_4,name_4 = raman_batch(directory,1200,2900,0.01)
    peaks = []
    signals = []
    for i in range(len(a_4)):
        for j in range(len(a_4[i])):
            for s in range(len(a[i])):
                if a_4[i][j] < a[i][s] + 2 and a_4[i][j] > a[i][s] - 2:
                    peaks.append(a_4[i][j])
                    signals.append(b_4[i][j])
    names = np.array(name_4)
    peaks = np.array(peaks)
    signals = np.array(signals)
    peaks = peaks.reshape(len(a),3)
    signals = signals.reshape(len(a),3)
    twod_g = signals[:,2]/signals[:,1]
    d_g = signals[:,0]/signals[:,1]
    for num in range(len(d_g)):
        if d_g[num] < filter:
            d_g[num]=0
    total = np.hstack((peaks,signals))
    total = np.column_stack((names,total,twod_g,d_g,c_3))
    np.savetxt(directory+'_peaks'+'.txt',total,fmt='%s')
    return total
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
    plt.xlabel("Raman shift, cm$^{-1}$", fontsize = 12)
