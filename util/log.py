"""Class for logging."""
import numpy as np
from visdom import Visdom


class Logger():
    """Logger for training."""

    def __init__(self, enable_visdom=False, curve_names=None):
        self.curve_names = curve_names
        if enable_visdom:
            self.vis = Visdom()
            assert self.vis.check_connection()
            self.curve_x = np.array([0])
        else:
            self.curve_names = None

    def log(self, xval=None, win_name='loss', **kwargs):
        """Log and print the information."""
        print("##############################################################")
        for key, value in kwargs.items():
            print(key, value, sep='\t')

        if self.curve_names:
            if not xval:
                xval = self.curve_x
            for i in range(len(self.curve_names)):
                name = self.curve_names[i]
                if name not in kwargs:
                    continue
                yval = np.array([kwargs[name]])
                self.vis.line(Y=yval, X=xval, win=win_name, update='append',
                              name=name, opts=dict(showlegend=True))
                                # name = name, opts = {'showlegend':True,'linecolor':np.array([[0,0,255],])})
                self.curve_x += 1

    def plot_curve(self, yvals, xvals, win_name='pr_curves'):
        """Plot curve."""
        self.vis.line(Y=np.array(yvals), X=np.array(xvals), win=win_name, opts={'linecolor':np.array([[0,0,0],])})
