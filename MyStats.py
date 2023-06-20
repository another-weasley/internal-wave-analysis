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

CURRENT_FILE = 'S09'

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
        fix_h = np.array(self.heights) / stat.mode(self.heights)
        fix_p = np.array(self.periods) / stat.mode(self.periods)
        self.slopes = fix_h / fix_p
        self.angles = []

    def show_height_hist(self):
        mean = stat.mean(self.heights)
        deviation = stat.stdev(self.heights)
        print('среднее высоты', round(mean, ndigits=2), 'стандартное отклонение высоты', round(deviation, ndigits=2))
        print('медиана высоты', stat.median(self.heights), 'мода высоты', stat.mode(self.heights))
        print(f"min: {min(self.heights)}, max: {max(self.heights)}")

        self.heights = np.array(self.heights)
        n = len(self.heights)
        df = pd.DataFrame({'h': self.heights, 'freq': np.zeros_like(self.heights)})
        df = df.groupby(['h']).count()
        df['rel_freq'] = df['freq'] / n

        x_data = list(df.index)
        y_data = list(df['rel_freq'])
        freq_data, y_data = df.to_numpy().transpose()

        k = math.ceil(math.sqrt(n))
        counts, bins = np.histogram(self.heights, bins=k, density=True)
        bins = bins[:-1]
        width_dict = {'S09': 0.2, 'S04': 0.0864, 'S02': 0.15, 'S05': 0.22, 'S07': 0.1}
        plt.bar(bins, counts, width=width_dict[CURRENT_FILE], edgecolor='black')
        shape, loc, scale = sst.lognorm.fit(self.heights)
        print(f"sigma={shape}, mu={scale}")
        x = np.linspace(min(x_data), max(x_data), 1000)
        dens = sst.lognorm.pdf(x, s=shape, loc=loc, scale=scale)
        plt.plot(x, dens, color='r')

        plt.title("Гистограмма высоты волн")
        plt.xlabel('H, м')
        plt.ylabel('Частота, []')
        plt.show()

        myPirson(list(counts), list(sst.lognorm.pdf(bins, s=shape, loc=loc, scale=scale)))

    def show_period_hist(self):
        mean = stat.mean(self.periods)
        deviation = stat.stdev(self.periods)
        print('среднее периода', mean, 'стандартное отклонение периода', deviation)
        print('медиана периода', stat.median(self.periods), 'мода периода', stat.mode(self.periods))
        print(f"min: {min(self.periods)}, max: {max(self.periods)}")

        self.periods = np.array(self.periods)
        n = len(self.periods)
        print(n)
        df = pd.DataFrame({'h': self.periods, 'freq': np.zeros_like(self.periods)})
        df = df.groupby(['h']).count()
        df['rel_freq'] = df['freq'] / n

        x_data = list(df.index)
        y_data = list(df['rel_freq'])
        freq_data, y_data = df.to_numpy().transpose()

        k = math.ceil(math.sqrt(n))
        counts, bins = np.histogram(self.periods, bins=k, density=True)
        bins = bins[:-1]
        width_dict = {'S09': 15, 'S04': 5, 'S02': 15, 'S05': 2.5, 'S07': 4.15}
        plt.bar(bins, counts, width=width_dict[CURRENT_FILE], edgecolor='black', align='edge')
        shape, loc, scale = sst.lognorm.fit(self.periods)
        print(f"sigma={shape}, mu={scale}")
        dens = sst.lognorm.pdf(x_data, s=shape, loc=loc, scale=scale)
        plt.plot(x_data, dens, color='r')

        plt.title("Гистограмма периода волн")
        plt.xlabel('T, минуты')
        plt.ylabel('Частота, []')
        plt.show()

        myPirson(list(counts), list(sst.lognorm.pdf(bins, s=shape, loc=loc, scale=scale)))

    def show_slope_hist(self):
        mean = stat.mean(self.slopes)
        deviation = stat.stdev(self.slopes)
        print('среднее крутизны', mean, 'стандартное отклонение крутизны', deviation)
        print('медиана периода', stat.median(self.slopes), 'мода периода', stat.mode(self.slopes))
        print(f"min: {min(self.slopes)}, max: {max(self.slopes)}")

        self.slopes = np.array(self.slopes)
        n = len(self.slopes)
        df = pd.DataFrame({'h': self.slopes, 'freq': np.zeros_like(self.slopes)})
        df = df.groupby(['h']).count()
        df['rel_freq'] = df['freq'] / n

        x_data = np.array(list(df.index)) + 0.00001
        y_data = list(df['rel_freq'])
        freq_data, y_data = df.to_numpy().transpose()

        k = math.ceil(math.sqrt(n))
        counts, bins = np.histogram(self.slopes, bins=k, density=True)
        bins = bins[:-1]
        width_dict = {'S09': 0.37, 'S04': 0.3, 'S02': 0.41, 'S05': 0.39, 'S07': 0.3}
        plt.bar(bins, counts, width=width_dict[CURRENT_FILE], edgecolor='black', align='edge')
        shape, loc, scale = sst.lognorm.fit(self.slopes)
        print(f"sigma={shape}, mu={scale}")
        dens = sst.lognorm.pdf(x_data, s=shape, loc=loc, scale=scale)
        plt.plot(x_data, dens, color='r')

        plt.title("Гистограмма крутизны волн")
        plt.xlabel('Крутизна, []')
        plt.ylabel('Частота, []')
        plt.show()

        myPirson(list(counts), list(sst.lognorm.pdf(bins, s=shape, loc=loc, scale=scale)))

    def show_angle_hist(self, isopic):
        derivative = np.diff(isopic)
        x = np.linspace(0, MINUTES, MINUTES * 6)
        x = x[:-1]
        angle = np.arctan(derivative)
        angle = np.rad2deg(angle)
        self.angles = [round(a, ndigits=2) for a in angle if a > 0]

        mean = stat.mean(self.angles)
        deviation = stat.stdev(self.angles)
        print('среднее крутизны', mean, 'стандартное отклонение крутизны', deviation)
        print('медиана периода', stat.median(self.angles), 'мода периода', stat.mode(self.angles))
        print(f"min: {min(self.angles)}, max: {max(self.angles)}")

        self.angles = np.array(self.angles)
        n = len(self.angles)
        df = pd.DataFrame({'h': self.angles, 'freq': np.zeros_like(self.angles)})
        df = df.groupby(['h']).count()
        df['rel_freq'] = df['freq'] / n

        x_data = np.array(list(df.index)) + 0.00001
        y_data = list(df['rel_freq'])
        freq_data, y_data = df.to_numpy().transpose()

        #k = math.ceil(math.sqrt(n))
        k = 40
        counts, bins = np.histogram(self.angles, bins=k, density=True)
        bins = bins[:-1]
        width_dict = {'S09': 0.12, 'S04': 0.12, 'S02': 0.1, 'S05': 0.2, 'S07': 0.1}
        plt.bar(bins, counts, width=width_dict[CURRENT_FILE], edgecolor='black', align='edge')
        shape, loc, scale = sst.lognorm.fit(self.angles)
        print(f"sigma={shape}, mu={scale}")
        dens = sst.lognorm.pdf(x_data, s=shape, loc=loc, scale=scale)
        plt.plot(x_data, dens, color='r')

        plt.title("Гистограмма амплитуды угла волнового склона")
        plt.xlabel('ɑ°')
        plt.ylabel('Частота, []')
        plt.show()

        myPirson(list(counts), list(sst.lognorm.pdf(bins, s=shape, loc=loc, scale=scale)))

    def show_spectral(self, isopic):
        x = np.linspace(0, MINUTES, MINUTES * 6)
        furier = fft.rfft(isopic)
        req = fft.rfftfreq(len(isopic))
        plt.plot(req, abs(furier))
        plt.show()