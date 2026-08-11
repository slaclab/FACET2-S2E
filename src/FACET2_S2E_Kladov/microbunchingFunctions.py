import math
from scipy.stats import moment
from scipy.ndimage import gaussian_filter1d

from Experimental_functions import *
from FACET2_S2E_Kladov.UTILITY_quickstart import *
from FACET2_S2E_Kladov.functionsForSims import make_a_plot

def make_modulated_bunch(beam, wavelength=30e-6, mod_amplitude=0.1, save_file=""):
    '''
    Add microbunching to the bunch
    '''
    indices_to_leave = []
    for (i,p) in enumerate(beam):
        if(np.random.rand()<1-0.5*mod_amplitude*(math.sin(2*math.pi*((p.t)[0])*3*10**8/wavelength)+1)):
            indices_to_leave.append(i)
            
    indices = np.array(indices_to_leave)
    P_local=beam[indices]
    P_local.charge = beam.charge
    if save_file!="":
        P_local.write(save_file+".h5")
    return P_local

## Display the bunch density distribution \rho(z)

def hist(data, label='Longitudinal Coordinate z (um)', num_bins=200, xlim=None):
    # Compute histogram (density=True normalizes area under curve to 1)
    hist_kwargs = dict(bins=num_bins, density=True)
    if xlim is not None:
        if len(xlim) != 2:
            raise ValueError('xlim must be a sequence of two values: (xmin, xmax)')
        hist_kwargs['range'] = xlim
    counts, bin_edges = np.histogram(data, **hist_kwargs)
    
    # Compute bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot as line (envelope)
    #plt.figure(figsize=(8, 4))
    #plt.plot(bin_centers, counts, color='black', linewidth=2)
    
    # Optional: Smooth the curve
    smoothed_counts = gaussian_filter1d(counts, sigma=1.0)

    fig = plt.figure(figsize=(8, 3.25))
    plt.plot(bin_centers, smoothed_counts, color='black')

    plt.fill_between(bin_centers, smoothed_counts, color='grey', alpha=0.5)
    
    if xlim is not None:
        plt.xlim(xlim)
    
    # Styling
    plt.xlabel(label, fontsize=12)
    #plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    plt.ylabel('Density', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()

## Spectrum functions

def get_spectrum(beam):
    nbins = 5000
    c = 2.99792458e8
    l_dist = (beam.t-np.mean(beam.t))*c
    sigz = moment(l_dist, moment=2) ** 0.5
    [hist, bin_edges] = np.histogram(l_dist, bins=nbins)
    x = np.zeros(nbins)
    for i in range(nbins):
        x[i] = (bin_edges[i]+bin_edges[i+1])/2
    fourier = np.fft.fft(hist)
    x = np.zeros(nbins)
    for i in range(nbins):
        x[i] = i/nbins
    return [bin_edges[1]-bin_edges[0],np.array([x, fourier]),sigz]


def print_spec(beam, maxlam, minlam):
    spec = get_spectrum(beam)
    bsize = spec[2]
    maxlam = np.minimum(maxlam, bsize/2.5)
    minFreq = spec[0]/maxlam
    maxFreq = spec[0]/minlam
    maxFreq = np.minimum(maxFreq, 0.5)
    specx = spec[1][0]
    specy = np.power(np.abs(spec[1][1]),2)
    idx_wl = find_nearest(specx,minFreq)
    idx_wh = find_nearest(specx,maxFreq)
    make_a_plot(specx[idx_wl:idx_wh],specy[idx_wl:idx_wh], label="Spectrum of z", x_label="Frequency (normalized)", y_label="Power", cartesian_axes=[False, True], axes_location=[0, 0])


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def analyze_spec(spec, maxlam, minlam):
    bsize = spec[2]
    maxlam = np.minimum(maxlam,bsize/2.5)
    minFreq = spec[0]/maxlam
    maxFreq = spec[0]/minlam
    maxFreq = np.minimum(maxFreq, 0.5)
    specx = spec[1][0]
    specy = np.power(np.abs(spec[1][1]),2)
    idx_wl = find_nearest(specx,minFreq)
    idx_wh = find_nearest(specx,maxFreq)
    idx_max = np.abs(specy[idx_wl:idx_wh]).argmax()+idx_wl

    idx_low = find_nearest(specx,np.maximum(minFreq, 0.8*specx[idx_max]))
    idx_high = find_nearest(specx, 1.2*specx[idx_max])

    idx_low = idx_wl
    idx_high = idx_wh
    
    l = idx_high-idx_low

    meanx = 0
    toty = 0
    for i in range(l):
        meanx += specx[idx_low+i]*specy[idx_low+i]
        toty += specy[idx_low+i]
    meanx = meanx/toty

    sigmax = 0
    for i in range(l):
        sigmax += np.power(specx[idx_low+i]-meanx,2)*specy[idx_low+i]
    sigmax = np.sqrt(sigmax/toty)

    idx_3sl = find_nearest(specx,meanx-3*sigmax)
    idx_3sh = find_nearest(specx,meanx+3*sigmax)
    l1 = idx_3sh-idx_3sl
    quadpower = 0
    for i in range(l1):
        quadpower += (specx[idx_3sl+i+1]-specx[idx_3sl+i])*specy[idx_3sl+i]
    return [meanx,sigmax,quadpower]


def get_microbunching_gain_from_beams(initial_beam, final_beam, lmax1, lmin1, lmax2, lmin2, file=""):
    ipars = analyze_spec(get_spectrum(initial_beam),lmax1,lmin1)
    fpars = analyze_spec(get_spectrum(final_beam),lmax2,lmin2)
    if file!="":
        np.savetxt(file,np.real(np.array([fpars,ipars])))
    return fpars[2]/ipars[2]

def get_microbunching_gain(initial_beam_path, final_beam_path, lmax1, lmin1, lmax2, lmin2, file=""):
    return get_microbunching_gain_from_beams(ParticleGroup(initial_beam_path), ParticleGroup(final_beam_path), lmax1, lmin1, lmax2, lmin2, file=file)