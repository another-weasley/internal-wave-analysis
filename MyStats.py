import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import statistics as stat
import math


def mylog(x, mu, sigma):
    return np.exp(-(np.log(x) / mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi))


def normalise_data(data):
    freq = [data.count(value) / len(data) for value in set(data)]
    return freq


class MyStat:
    def __init__(self, waves):
        self.waves = waves
        self.periods = [wave.T for wave in self.waves]


    def show_height_hist(self):
        heigths = [wave.height() for wave in self.waves if wave.height() > 0]
        height_mean = stat.mean(heigths)
        height_deviation = stat.stdev(heigths)
        print(height_mean, height_deviation)
        norm_heights = normalise_data(heigths)
        dens = mylog(np.array(heigths), 0.3, 0.3)

        x = np.array(list(sorted(set(heigths))))
        popt, pcov = curve_fit(mylog, x, norm_heights)
        print(popt)
        # dens = lognorm.pdf(sorted(heigths), s=height_deviation, scale=height_mean)
        # print(scipy.stats.chisquare(norm_heights, dens))

        data = heigths
        # гистограмма высоты
        plt.clf()
        data = heigths
        plt.hist(data, edgecolor='black', weights=np.ones_like(data) / len(data),
                 bins=math.ceil(math.sqrt(len(heigths))))
        # plt.plot(data, dens, linewidth=2, color='r')
        # plt.hist(sorted(set(heigths)), edgecolor='black', weights=norm_heights, bins=math.ceil(math.sqrt(len(heigths))))
        # plt.hist(np.array(sorted(list(set(heigths)))), density=True, bins=math.ceil(math.sqrt(len(heigths))))
        plt.plot(data, np.ones_like(data) / len(data), '-', color='green')
        plt.title("Гистограмма относительных частот высоты волн")
        plt.xlabel('H, м')
        plt.ylabel('Частота, []')

        # plt.plot(sorted(heigths), dens, linewidth=2, color='r')
        m, s = popt[0], popt[1]
        print(type(m))
        # plt.plot(x, mylog(x, m, s), color='r')

        dens = mylog(np.array(sorted(list(set(heigths)))), sigma=height_deviation, mu=height_mean)
        plt.plot(sorted(list(set(heigths))), dens, linewidth=2, color='r')

        plt.show()

    def show_period_hist(self):
        period_mean = stat.mean(self.periods)
        period_deviation = stat.stdev(self.periods)
        print(period_mean, period_deviation)
        norm_periods = normalise_data(self.periods)
        dens = mylog(np.array(sorted(list(set(self.periods)))), sigma=period_deviation, mu=period_mean)

        # гистограмма периода
        plt.hist(sorted(set(self.periods)), edgecolor='black', weights=norm_periods,
                 bins=math.ceil(math.sqrt(len(self.periods))))
        plt.title("Гистограмма относительных частот периода волн")
        plt.xlabel('T, минуты')
        plt.ylabel('Частота, []')
        plt.plot(sorted(list(set(self.periods))), dens, linewidth=2, color='r')
        plt.show()
