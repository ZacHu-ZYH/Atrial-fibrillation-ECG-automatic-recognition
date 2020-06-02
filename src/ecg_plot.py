import matplotlib.pyplot as plt 
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import AutoMinorLocator
import numpy as np
# =================================================================================
def ecg_plot(ecg,i):
    fs = 128
    Time=np.linspace(0, len(ecg)/fs, num=len(ecg))
  
    fig, ax = plt.subplots(figsize=(16,5))
    ax.plot(Time,ecg,'-', lw=1.0, color='k')

    ax.set_xticks(np.arange(0,10,0.2),)  
    plt.xticks( rotation='vertical')  
    ax.set_yticks(np.arange(-1,1,0.03))

    ax.minorticks_on()

    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    ax.grid(which='minor', linestyle='-', linewidth='0.5', color=(1, 0.7, 0.7))

    ax.set_ylim(-0.3, 0.4)
    ax.set_xlim(0, 10)
    plt.ylabel('ECG0(mV)')
    plt.xlabel('time(s)')
    # plt.title('Abnormal Record')
#    plt.show()
    plt.text(0,0.3,'19210720196huzhaoyu',weight="bold", color="b") 
    plt.savefig('%d.png'%i)