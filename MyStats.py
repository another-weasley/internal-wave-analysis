import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import statistics as stat
import math
import scipy.stats
from scipy.interpolate import interp1d

def mylog(x, mu, sigma):
    return np.exp(-(np.log(x) / mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi))


def myPirson(observed, expected):
    n = len(observed)
    chi2_arr = [((observed[i] - expected[i])**2) / expected[i] * n for i in range(0, n)]
    chi2 = sum(chi2_arr)
    dof = n - 2 - 1  # число степеней свободы
    chi_kr = scipy.stats.chi2.ppf(1-0.05, dof)
    print('х_набл', chi2, 'хи_крит', chi_kr)


# def myexp(x, mu):
#     x = np.array(x)
#     l = 1 / mu
#     return l*np.exp(-l*x)


def normalise_data(data):
    freq = [data.count(value) / len(data) for value in sorted(list(set(data)))]
    return freq


class MyStat:
    def __init__(self, waves):
        self.waves = [wave for wave in waves if wave.height() >= 0.1]
        self.periods = [round(wave.T, ndigits=4) for wave in self.waves if wave.T > 2]
        self.heights = [round(wave.height(), ndigits=10) for wave in self.waves]

    def show_height_hist(self):
        height_mean = stat.mean(self.heights)
        height_deviation = stat.stdev(self.heights)
        print('среднее высоты: ', height_mean, 'стандартное отклонение высоты: ', height_deviation)
        norm_heights = normalise_data(self.heights)

        # гистограмма высоты
        plt.hist(sorted(set(self.heights)), edgecolor='black', weights=norm_heights, bins=math.ceil(math.sqrt(len(self.heights))))
        counts, bins = np.histogram(list(sorted(set(self.heights))), bins=math.ceil(math.sqrt(len(self.heights))), weights=norm_heights)

        plt.title("Гистограмма относительных частот высоты волн")
        plt.xlabel('H, м')
        plt.ylabel('Частота, []')

        x = np.array(list(sorted(self.heights)))
        last_bins = [bins[i+1] for i in range(0, len(bins)-1)]
        bins = bins[:-1]
        counts = counts + 0.0000001
        # popt, _ = curve_fit(mylog, counts, last_bins)
        # print(popt)
        # m, s = popt[0], popt[1]
        # plt.plot(x, mylog(np.array(x), m, s), color='red')
        # dens = mylog(np.array(last_bins), 2.3, 1)
        # plt.show()

        plt.hist(sorted(set(self.heights)), edgecolor='black', weights=norm_heights,
                 bins=math.ceil(math.sqrt(len(self.heights))))
        densf = interp1d(bins, counts, kind='quadratic', bounds_error=False, fill_value='extrapolate')
        new_x = np.linspace(min(self.heights), max(self.heights), 100)
        dens = densf(new_x)
        plt.plot(new_x, dens, color='orange')
        popt, _ = curve_fit(mylog, new_x, dens)
        print(popt)
        m, s = popt[0], popt[1]
        plt.plot(new_x, mylog(np.array(new_x), m, s), color='purple')
        plt.show()
        #myPirson(np.array(counts), np.array(dens))

    def show_period_hist(self):
        period_mean = stat.mean(self.periods)
        period_deviation = stat.stdev(self.periods)
        print('среднее периода', period_mean, 'стандартное отклонение периода', period_deviation)
        norm_periods = normalise_data(self.periods)

        # гистограмма периода
        plt.hist(sorted(set(self.periods)), edgecolor='black', weights=norm_periods,
                 bins=math.ceil(math.sqrt(len(self.periods))))
        counts, bins = np.histogram(list(sorted(set(self.periods))), bins=math.ceil(math.sqrt(len(self.periods))),
                                    weights=norm_periods)
        x = np.array(list(sorted(self.periods)))
        last_bins = [bins[i + 1] for i in range(0, len(bins) - 1)]
        counts = counts + 0.0000001
        popt, _ = curve_fit(mylog, counts, last_bins)
        print(popt)
        m, s = popt[0], popt[1]
        plt.plot(x, mylog(np.array(x), m, s), color='red')
        dens = mylog(np.array(last_bins), m, s)
        myPirson(np.array(counts), np.array(dens))

        plt.title("Гистограмма относительных частот периода волн")
        plt.xlabel('T, минуты')
        plt.ylabel('Частота, []')
        plt.show()
