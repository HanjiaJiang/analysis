import numpy as np
import matplotlib.pyplot as plt

def hist_spectral(hist, bin_width, given_freq_resol):
    # bin_width: in second
    # given_freq_resol: in Hz

    # limits of frequency resolutions for output
    freq_resol_lower_limit = 1/(len(hist) * bin_width)  # lower limit = 1/time length of samples
    freq_resol_upper_limit = 1/(bin_width*2)            # upper limit = sampling rate/2
    if given_freq_resol < freq_resol_lower_limit or given_freq_resol > freq_resol_upper_limit:
        print('frequency resolution out of range; force using the limit')
        given_freq_resol = freq_resol_lower_limit

    # calculation of psd
    # assign the length of output according to given_freq_resol
    psd = np.fft.fft(hist, n=len(hist) * freq_resol_lower_limit / given_freq_resol)
    psd_positive = psd[0:int(len(psd)/2)]
    psd_negative = psd[int(len(psd)/2):-1]
    freq_list = np.linspace(0.0, (len(psd_positive)-1)*given_freq_resol, len(psd_positive))
    return psd_positive, psd_negative, freq_list


# test
sin1 = np.sin(np.arange(0.0, 200*np.pi, 0.1*np.pi)+1)+1
sin2 = np.sin(2*np.arange(0.0, 200*np.pi, 0.1*np.pi)+1)+1
sin3 = np.sin(3*np.arange(0.0, 200*np.pi, 0.1*np.pi)+1)+1

data = sin1 + sin2 + sin3
# spectrum = np.fft.fft(data)
spectrum_p, spectrum_n, freqs = hist_spectral(data, 0.001, 5.0)

fig1, axs1 = plt.subplots(2, 1)
axs1 = axs1.ravel()
axs1[0].plot(data)
# axs1[1].plot(spectrum)
axs1[1].plot(freqs, spectrum_p)
print(freqs)
plt.show()

