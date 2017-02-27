from __future__ import print_function

from .modeler import FastTemplateModeler
from .template import Template
from .old_utils import approximate_template, rms_resid_over_rms_fast
import gatspy.datasets.rrlyrae as rrl
import os

try:
    # Python 2.x only
    import cPickle as pickle
except ImportError:
    import pickle



def get_rrlyr_templates(template_fname=None, errfunc=rms_resid_over_rms_fast,
                        stop=None, filts='r', nharmonics=6, redo=False):
    """
    read templates for a given filter(s) from
    the gatspy.datasets.rrlyrae package, approximate each
    template with the minimal number of harmonics such
    that `errfunc(Cn, Sn, T, Y)` < `stop` and save values
    in a pickled file given by template_fname if template_fname
    is not None.
    """

    # Obtain RR Lyrae templates
    templates = rrl.fetch_rrlyrae_templates()

    # Select the right ID's
    IDs = [ t for t in templates.ids if t[-1] in list(filts) ]

    # Get (phase, amplitude) data for each template
    Ts, Ys = zip(*[ templates.get_template(ID) for ID in IDs ])

    ftp_templates = None
    if not template_fname is None \
          and os.path.exists(template_fname) \
          and not redo:
        ftp_templates = pickle.load(open(template_fname, 'rb'))
    else:
        #print "loading ftp_templates"
        ftp_templates = { ID : Template(phase=T, y=Y, errfunc=errfunc,
                                        nharmonics=nharmonics, stop=stop).precompute() \
                                    for ID, T, Y in zip(IDs, Ts, Ys) }
        #print "done"
        if not template_fname is None:
            pickle.dump(ftp_templates, open(template_fname, 'wb'))

    return ftp_templates


class FastRRLyraeTemplateModeler(FastTemplateModeler):
    """
    RR Lyrae Template modeler

    Parameters
    ----------
    x: np.ndarray, list
        independent variable (time)
    y: np.ndarray, list
        array of observations
    err: np.ndarray
        array of observation uncertainties
    filts: str (default: 'r')
        string containing one or more of 'ugriz'
    loud: boolean (default: True), optional
        print status
    ofac: float, optional (default: 10)
        oversampling factor -- higher values of ofac decrease
        the frequency spacing (by increasing the size of the FFT)
    hfac: float, optional (default: 1)
        high-frequency factor -- higher values of hfac increase
        the maximum frequency of the periodogram at the
        expense of larger frequency spacing.
    stop: float, optional (default: 2E-2)
        will pick minimum number of harmonics such that
        rms(trunc(template) - template) / rms(template) < stop
    nharmonics: None or int, optional (default: None)
        Keep a constant number of harmonics
    template_fname: str, optional
        Filename to load/save template
    errfunc: callable, optional (default: rms_resid_over_rms)
        A function returning some measure of error resulting
        from approximating the template with a given number
        of harmonics
    redo : bool (optional)
        Recompute templates even if they are saved

    """
    def __init__(self, filts='ugriz', redo=False, **kwargs):
        FastTemplateModeler.__init__(self, **kwargs)
        self.filts = filts
        self.params['redo'] = redo
        self.params['filts'] = self.filts
        self._load_templates()

    def _load_templates(self):
        pars = [ 'filts', 'template_fname', 'errfunc', 'stop', 'nharmonics', 'redo' ]
        kwargs = { par : self.params[par] for par in pars if par in self.params }
        self.templates = get_rrlyr_templates(**kwargs)
