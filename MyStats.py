import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import statistics as stat
import math
import scipy.stats as sst
from scipy.interpolate import interp1d
import pandas as pd
import seaborn as sns
from SeaPoint import MINUTES
from scipy.interpolate import interp1d
import scipy.fft as fft

CURRENT_FILE = 'S04'
def mylog(x, mu, sigma):
    return np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi))

def myPirson_old(observed, expected, N):
    n = len(observed)
    chi2_arr = [((observed[i] - expected[i])**2) / expected[i] for i in range(0, n)]
    chi2 = sum(chi2_arr)
    dof = n - 2 - 1  # число степеней свободы
    chi_kr = sst.chi2.ppf(1-0.05, dof)
    print('х_набл', chi2, 'хи_крит', chi_kr)


def myPirson(observed, expected):
    n = len(observed)
    chi2_arr = [((observed[i] - expected[i])**2) / expected[i] for i in range(0, n)]
    chi2 = sum(chi2_arr)
    dof = n - 2 - 1  # число степеней свободы
    chi_kr = sst.chi2.ppf(1-0.05, dof)
    print('х_набл', chi2, 'хи_крит', chi_kr)


def normalise_data(data):
    freq = [data.count(value) / len(data) for value in sorted(list(set(data)))]
    return freq


class MyStat:
    def __init__(self, waves):
        self.waves = [wave for wave in waves if wave.height() >= 0.5]
        self.periods = [round(wave.T) for wave in self.waves if wave.T > 0]
        self.heights = [round(wave.height(), ndigits=2) for wave in self.waves]
        self.slopes = [round(wave.slope(), ndigits=2) for wave in waves if wave.slope() > 0]

    def show_height_hist(self):
        mean = stat.mean(self.heights)
        deviation = stat.stdev(self.heights)
        print('среднее высоты', mean, 'стандартное отклонение высоты', deviation)

        self.heights = np.array(self.heights)
        n = len(self.heights)
        df = pd.DataFrame({'h': self.heights, 'freq': np.zeros_like(self.heights)})
        df = df.groupby(['h']).count()
        df['rel_freq'] = df['freq'] / n

        x_data = list(df.index)
        y_data = list(df['rel_freq'])
        freq_data, y_data = df.to_numpy().transpose()

        # popt, _ = curve_fit(mylog, x_data, y_data)
        # print(popt)
        # m, s = popt[0], popt[1]
        # dens = mylog(np.array(x_data), m, s)
        # plt.hist(x_data, weights=y_data, bins=n, color='gray', edgecolor='gray')
        # plt.plot(x_data, dens, color='red')
        #
        # myPirson(list(y_data), list(mylog(np.array(x_data), m, s)), int(sum(freq_data)))
        #
        # plt.title("Гистограмма относительных частот высоты волн")
        # plt.xlabel('H, м')
        # plt.ylabel('Частота, []')
        # plt.show()

        k = math.ceil(math.sqrt(n))
        counts, bins = np.histogram(self.heights, bins=k, density=True)
        bins = bins[:-1]
        width_dict = {'S09': 0.2, 'S04': 0.0865}
        plt.bar(bins, counts, width=width_dict[CURRENT_FILE], edgecolor='black')
        shape, loc, scale = sst.lognorm.fit(self.heights)
        print(f"sigma={shape}, mu={scale}")
        dens = sst.lognorm.pdf(x_data, s=shape, loc=loc, scale=scale)
        plt.plot(x_data, dens, color='r')

        plt.title("Гистограмма высоты волн и ее аппроксимация логнормальным распределением")
        plt.xlabel('H, м')
        plt.ylabel('Частота, []')
        plt.show()

        myPirson(list(counts), list(sst.lognorm.pdf(bins, s=shape, loc=loc, scale=scale)))

    def show_period_hist(self):
        mean = stat.mean(self.periods)
        deviation = stat.stdev(self.periods)
        print('среднее периода', mean, 'стандартное отклонение периода', deviation)

        self.periods = np.array(self.periods)
        n = len(self.periods)
        df = pd.DataFrame({'h': self.periods, 'freq': np.zeros_like(self.periods)})
        df = df.groupby(['h']).count()
        df['rel_freq'] = df['freq'] / n

        x_data = list(df.index)
        y_data = list(df['rel_freq'])
        freq_data, y_data = df.to_numpy().transpose()

        plt.bar(x_data, y_data, width=1, color='gray')
        shape, loc, scale = sst.lognorm.fit(self.periods)
        print(f"sigma={shape}, mu={scale}")
        dens = sst.lognorm.pdf(x_data, s=shape, loc=loc, scale=scale)
        plt.plot(x_data, dens, color='r')

        myPirson(list(y_data), list(dens), len(self.periods))

        plt.title("Гистограмма относительных частот периода волн")
        plt.xlabel('T, минуты')
        plt.ylabel('Частота, []')
        plt.show()

    def show_slope_hist(self):
        mean = stat.mean(self.slopes)
        deviation = stat.stdev(self.slopes)
        print('среднее высоты', mean, 'стандартное отклонение высоты', deviation)

        self.slopes = np.array(self.slopes)
        n = len(self.slopes)
        df = pd.DataFrame({'h': self.slopes, 'freq': np.zeros_like(self.slopes)})
        df = df.groupby(['h']).count()
        df['rel_freq'] = df['freq'] / n

        x_data = np.array(list(df.index)) + 0.00001
        y_data = list(df['rel_freq'])
        freq_data, y_data = df.to_numpy().transpose()

        k = math.ceil(math.sqrt(n))
        print(y_data)
        counts, bins = np.histogram(self.slopes, bins=k, density=True)
        bins = bins[:-1]
        plt.bar(bins, counts, width=0.0055, edgecolor='black')
        shape, loc, scale = sst.lognorm.fit(self.slopes)
        print(f"sigma={shape}, mu={scale}")
        dens = sst.lognorm.pdf(x_data, s=shape, loc=loc, scale=scale)
        plt.plot(x_data, dens, color='r')
        plt.show()

        myPirson(list(y_data), list(dens), len(self.slopes))

    def show_spectral(self, isopic):
        x = np.linspace(0, MINUTES, MINUTES * 6)
        furier = fft.rfft(isopic)
        req = fft.rfftfreq(len(isopic))
        plt.plot(req, abs(furier))
        plt.show()