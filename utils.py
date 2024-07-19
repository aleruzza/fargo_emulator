import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator, interp1d

def generate_ic(slopes):
    #generating initial conditions
    slopes = np.array(slopes)
    x = np.linspace(-3, 3, 128)
    y = np.linspace(-3, 3, 128)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2+yy**2)
    ict = np.float32(r**(-slopes.reshape(-1,1,1))*((r<3) & (r>0.4)))
    
    ict = np.log10(1+np.float32(ict))
    ict = np.expand_dims(ict, axis=1)
    return torch.tensor(ict, dtype=torch.float32)


def generate_sym(r, prof):
    #generating initial conditions
    x = np.linspace(-3, 3, 128)
    y = np.linspace(-3, 3, 128)
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2+yy**2)
    f = interp1d(r, prof, fill_value=0, bounds_error=False)
    
    return f(rr)
    

def norm_labels(labels):
    ''' Normalizes labels in the correct format that should input to the emulator.
    	Provide the unnormalized labels in shape (N, 5) with the parameters in axis=1 in the
    	following order:
        PlanetMass, AspectRatio, Alpha, InvStokes1, FlaringIndex
    '''
    labels = np.array(labels).reshape(-1, 5)
    max = np.array([1e-2, 0.1, 0.01, 1e3, 0.35])
    min = np.array([1e-5, 0.03, 1e-4, 10, 0])
    for i in [0, 2, 3]:
        labels[:, i] = np.log10(labels[:,i])
        max[i] = np.log10(max[i])
        min[i] = np.log10(min[i])
    labels = 2*(labels-min)/(max-min) - 1
    return torch.tensor(labels, dtype=torch.float32)
    
    
def getprofile(data, x, y, r, mode='cyl'):
    if mode=='cart':
        rho = RegularGridInterpolator((x, y), data, bounds_error=False, method='cubic')
        theta = np.linspace(-np.pi, np.pi, 300)
        rr, tt = np.meshgrid(r, theta)
        xx = rr*np.cos(tt)
        yy = rr*np.sin(tt)
        datacyl = np.nan_to_num(rho((xx, yy)))
        return datacyl.mean(axis=0)
    else:
        return data.mean(axis=0)
