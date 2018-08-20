import numpy as np
import myplotlibrary4_3 as myplt
from scipy.optimize import curve_fit
from scipy.special import erf


def winprob_fit(v1, path, v2=None, nbins=64, normed_=True,
                xlab='Price (% list)', ylab='Count (normalized)',
                plot_two_=True, file1='temp', file2='temp2'):

    def cumu(x, sigma, mu):
        return 0.5 * (1 + erf((x - mu) / (sigma * np.sqrt(2))))

    def func(x, sigma_, mu_):
        return np.exp(-0.5 * ((x - mu_) / sigma_) ** 2) / (sigma_ *
                                                           np.sqrt(2 * np.pi))

    count, bins = np.histogram(v1, bins=64, normed=True)
    count_cum = np.cumsum(count)

    # Shift horizontal axis coordinate by one half the bin width
    bins_ = (bins + np.diff(bins, 1)[0] / 2)[:-1]

    popt, pcov = curve_fit(func, bins_, count)

    x_hat = np.arange(bins_.min(), bins_.max(), 0.01)
    y_hat = func(x_hat, popt[0], popt[1])
    y_hat_cum = cumu(x_hat, popt[0], popt[1])  # np.cumsum(y_hat)

    myplt.scat(bins_, count, path,
               xlabel1=xlab, ylabel1=ylab, plot_two=plot_two_,
               x2=x_hat, y2=y_hat, filename=file1)

    # myplt.scat(bins_[::-1], count_cum / count_cum.max(), path,
    #            xlabel1=xlab, ylabel1='Win rate', plot_two=plot_two_,
    #            x2=x_hat[::-1], y2=y_hat_cum, filename=file2)

    myplt.scat(bins_, count_cum / count_cum.max(), path,
               xlabel1=xlab, ylabel1='1 - win rate', plot_two=plot_two_,
               x2=x_hat, y2=y_hat_cum, filename=file2)

    return bins_, count
