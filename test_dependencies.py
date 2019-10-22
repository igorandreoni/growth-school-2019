"""
This code is for testing that all dependencies for the GROWTH
school are successfully installed and can be imported.

If one of these import lines fails, exit and attempt to install it at the 
command-line by running:

$ pip install <missing_module>

and trying to run this code again.  If the module is present, but still 
causing problems, it may be an old version, so try updating it:

$ pip install --upgrade <old_module>

If one of the external dependencies fails, consult the README.md file for
how to install it properly.  Finally, if all else fails, try a simple google 
search with your error or resort to the backup plan of downloading the
VirtualBox image with all dependencies already installed (see README.md).
"""

def test_dependency(dep, alternate_name=None):
    """
    Test external dependency by trying to run it at the command line.
    """
    try:
        subprocess.check_output(dep, stderr=subprocess.PIPE, shell=True)
        print("%s is installed properly as %s. OK" % (dep, dep))
        return 1
    except subprocess.CalledProcessError:
        try:
            subprocess.check_output(alternate_name, stderr=subprocess.PIPE, shell=True)
            print("%s is installed properly as %s. OK" % (dep, alternate_name))
            return 1
        except subprocess.CalledProcessError:
            print("===%s/%s IS NOT YET INSTALLED PROPERLY===" % (dep, alternate_name))
            return 0

if __name__ == '__main__':
    print("Testing Python Dependencies")
    
    # 1: Python Basics
    import astropy.table
    from matplotlib import pyplot as plt
    import math
    import astropy.coordinates
    import astropy.units as u
    import astropy.io.fits
    import astropy.stats
    import astropy.table
    import astropy.wcs
    import astropy.cosmology
    import scipy.optimize
    import scipy.odr
    from astroquery.irsa import Irsa
    print("1: Python Basics: Python Dependencies Installed Successfully")
    
    # 2: Image Reduction
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.stats import sigma_clipped_stats
    import glob
    import os
    import subprocess
    import warnings
    import numpy as np
    import matplotlib.pyplot as plt
    import photutils
    import pyregion
    print("2: Image Reduction: Python Dependencies Installed Successfully")
    
    # 3:  Photometry
    import numpy as np
    import numpy.ma as ma
    import os
    import astropy.units as u
    from astropy.io import ascii
    from astropy.coordinates import SkyCoord
    from astropy.wcs import WCS
    from astropy.stats import sigma_clipped_stats, sigma_clip
    import subprocess
    from astropy.io import fits
    import matplotlib.pyplot as plt
    import os
    from astroquery.vizier import Vizier
    from astropy.table import Table
    from astropy.table import Table
    from astropy import units as u
    from photutils import SkyCircularAperture, SkyCircularAnnulus, aperture_photometry
    print("3: Photometry: Python Dependencies Installed Successfully")
    
    # 4:  Observing Run Preparation
    import numpy as np
    from astropy import units as u
    from astropy.time import Time
    from astropy.coordinates import SkyCoord
    from astropy.coordinates import EarthLocation
    import pytz
    from astroplan import Observer, FixedTarget
    from astropy.utils.iers import conf
    from astroplan import download_IERS_A 
    from astropy.coordinates import get_sun, get_moon, get_body
    from astroplan import moon_illumination
    import matplotlib.pyplot as plt
    from astroplan.plots import plot_sky, plot_airmass
    from astroplan.plots import plot_finder_image
    from astroquery.skyview import SkyView
    from astroplan import AltitudeConstraint, AirmassConstraint, AtNightConstraint, MoonSeparationConstraint
    from astroplan import is_observable, is_always_observable, months_observable
    from astropy.table import Table
    from astroplan import observability_table
    from astropy.io import ascii
    print("4: Observing Run Preparation: Python Dependencies Installed Successfully")
    
    # 5:  Image Subtraction
    import numpy as np
    from astropy.io import fits #FITS files handling
    import os  #Call commands from outside Python
    from astropy.io import ascii  #Read/write ascii files
    from scipy.signal import convolve as scipy_convolve
    from astropy.convolution import convolve 
    from astropy.convolution import Gaussian2DKernel
    from astropy.stats import sigma_clipped_stats
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from astropy.stats import SigmaClip
    from photutils import Background2D, MedianBackground
    from photutils import make_source_mask
    import matplotlib.pyplot as plt
    import subprocess
    import shutil
    from astropy.stats import SigmaClip
    from photutils import Background2D, MedianBackground
    from photutils import make_source_mask
    from image_registration import chi2_shift
    from image_registration.fft_tools import shift
    import scipy
    from scipy import ndimage, misc
    import matplotlib.pyplot as plt
    import numpy.fft
    from image_registration import chi2_shift
    from image_registration.fft_tools import shift
    import scipy
    from scipy import ndimage, misc
    import matplotlib.pyplot as plt
    import numpy.fft
    print("5: Image Subtraction: Python Dependencies Installed Successfully")
    
    # 6:  Gravitational Wave Localization
    import astropy.utils.data
    from matplotlib import pyplot as plt
    import numpy as np
    import healpy as hp
    from astropy.table import Table, vstack, hstack, Column
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    import ligo.skymap.plot
    from scipy.stats import norm
    import scipy.stats
    print("6: Gravitational Wave Localization: Python Dependencies Installed Successfully")
    
    # 7:  Machine Learning
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import io
    import pydotplus
    from IPython.display import Image
    from sklearn.model_selection import train_test_split
    from sklearn import tree
    from sklearn.metrics import confusion_matrix
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    print("7: Machine Learning: Python Dependencies Installed Successfully")
    
    # 8:  Spectroscopy
    import numpy as np
    from astropy.io import fits
    from astropy.stats import sigma_clip
    from scipy.interpolate import interp1d
    from scipy.interpolate import LSQBivariateSpline, LSQUnivariateSpline
    from scipy.optimize import fmin
    from scipy.optimize import least_squares
    import pickle
    import matplotlib.pylab as plt
    import os
    import subprocess
    from scipy.interpolate import LSQUnivariateSpline as real_LSQUnivariateSpline
    print("8: Spectroscopy: Python Dependencies Installed Successfully")
    
    # 9:  Lightcurve Analysis
    import numpy as np
    from astropy.stats import LombScargle
    import matplotlib.pyplot as plt
    import os
    print("9: Lightcurve Analysis: Python Dependencies Installed Successfully")
    
    # 10: Asteroids
    import requests
    import numpy as np
    import pandas as pd
    import glob
    import matplotlib.pyplot as plt
    import os
    import tarfile
    from astroquery.jplhorizons import Horizons # if fail, try upgrading astroquery with # pip install --upgrade astroquery
    import datetime
    from astropy.time import Time
    from IPython.display import Image as Image_jupyter
    from IPython.display import display
    import pdb
    import warnings
    from io import StringIO
    import tensorflow as tf
    from PIL import Image, ImageOps
    print("10: Asteroids: Python Dependencies Installed Successfully")
    
    # 11: Xray Analysis
    from astropy.io import fits
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    import scipy as sc
    import os
    import numpy.fft as fft
    print("11: Xray Analysis: Python Dependencies Installed Successfully")
    
    # 12: Radio Analysis
    import numpy as np
    import matplotlib.pyplot as plt
    import emcee
    from astropy.io import ascii
    import corner
    import os
    from timeit import default_timer as timer
    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    from scipy.optimize import least_squares, curve_fit
    from scipy.stats import f
    print("12: Radio Analysis: Python Dependencies Installed Successfully")
    
    # Test external program dependencies
    print("\nTesting External Dependencies")
    dependencies = [('sextractor', 'sex'), ('SWarp', 'swarp'), ('psfex', 'PSFEx'), ('ds9', 'DS9')]
    i = 0
    for dep_name1, dep_name2 in dependencies:
        i += test_dependency(dep_name1, dep_name2)
    print("%i out of %i external dependencies installed properly.\n" % (i, len(dependencies)))
    if i != len(dependencies):
        print("Please correctly install these programs before continuing by following the instructions in README.md.")
    else:
        print("You are have successfully installed all dependencies.  Congratulations!")
