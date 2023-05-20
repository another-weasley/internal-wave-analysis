import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import gsw
import math
from functools import lru_cache
import seaborn as sns
import statistics as stat
import scipy.stats

class Wave:
    def __init__(self, min1, max, min2, T):
        self.min1 = min1
        self.max = max
        self.min2 = min2
        self.T = T * 10 / 60 # т.е. в минутах

    def height(self):
        return (2*self.max - self.min1 - self.min2) / 2

class SeaPoint:

    def __init__(self, file):
        # парсим данные из mat-файла и приводим их к numpy виду
        mat = sio.loadmat(file)
        temp_header = list(mat.keys())[3]
        height_data_header = list(mat.keys())[8]
        height_header = list(mat.keys())[6]

        self.temp_data = np.array(mat[temp_header])
        self.temp_data = self.temp_data.transpose()
        self.height_data = np.array(mat[height_data_header])[0]
        self.latitude = 42
        self.longitude = 131
        self.height = mat[height_header][0][0]
        # print(mat["cS09"])

        self.salinity_woa = {5: 33.035, 10: 33.292, 15: 33.363, 20: 33.435, 25: 33.492,
                         30: 33.532, 35: 33.590, 40: 33.641, 45: 33.712, 50: 33.761,
                         55: 33.830, 60: 33.868, 65: 33.902, 70: 33.929, 75: 33.954,
                         80: 33.983, 85: 33.995, 90: 34.003, 95: 34.012, 100: 34.018}
        self.salinity = []
        for h in self.height_data:
            for key, value in self.salinity_woa.items():
                if abs(h-key) <= 2.5:
                    self.salinity.append(value)
                    break
        self.salinity = np.array(self.salinity)
    def show_salinity_profile(self):
        keys = np.array(list(self.salinity_woa.keys()))
        values = np.array(list(self.salinity_woa.values()))
        plt.gca().invert_yaxis()
        plt.plot(values, keys)
        plt.title("Профиль солености")
        plt.xlabel('S, %')
        plt.ylabel('z, м')
        plt.show()

    def show_temp(self):
        fig, ax = plt.subplots(constrained_layout=True)
        y = self.height_data
        dt = np.shape(self.temp_data)[1]  # число измерений
        x = np.linspace(0, 14, dt)
        X, Y = np.meshgrid(x, y)
        plt.gca().invert_yaxis()
        Z = self.temp_data
        cs = ax.contourf(X, Y, Z, cmap=plt.cm.jet)

        # add continuous color bar
        norm = matplotlib.colors.Normalize(vmin=cs.cvalues.min(), vmax=cs.cvalues.max())
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cs.cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ticks=cs.levels)
        cbar.ax.set_ylabel("температура, С")

        ax.set_title("Поле температуры")
        ax.set_xlabel("t, дни")
        ax.set_ylabel("z, м")
        plt.show()

    @lru_cache()
    def show_potential_density(self):
        fig, ax = plt.subplots(constrained_layout=True)
        y = self.height_data
        dt = np.shape(self.temp_data)[1]  # число измерений
        x = np.linspace(0, 14, dt)
        X, Y = np.meshgrid(x, y)
        plt.gca().invert_yaxis()
        self.height_data = -1 * self.height_data  # потому что вниз, а не вверх (иначе беды с давлением)
        pressure = gsw.p_from_z(self.height_data, self.latitude)
        absolute_salinity = gsw.SA_from_SP(self.salinity, pressure, self.longitude, self.latitude)
        # приведем аболютную соленость к такому же виду, в каком лежит температура (была ошибка)
        abs_sal_arr = np.zeros((len(self.height_data), 120961))
        for i in range(0, len(self.height_data), 1):
            abs_sal_arr[i, :] = absolute_salinity[i]
        density_data = gsw.density.sigma0(abs_sal_arr, self.temp_data) + 1000
        cs = ax.contourf(X, Y, density_data, cmap=plt.cm.jet)
        norm = matplotlib.colors.Normalize(vmin=cs.cvalues.min(), vmax=cs.cvalues.max())
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cs.cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ticks=cs.levels)
        cbar.ax.set_ylabel("плотность, кг/м^3")
        ax.set_title("Потенциальная плотность")
        ax.set_xlabel("t, дни")
        ax.set_ylabel("z, м")
        #plt.show()

        # вывести начальный и конечный срез плотности
        #print("t_0\n", density_data[:, 0])
        #print("t_end\n", density_data[:, len(density_data[0])-1])
        # for i in range(0, len(self.height_data)):
        #     print(-self.height_data[i], density_data[i, 0], density_data[i, len(density_data[0])-1])
        return density_data
    @lru_cache()
    def find_isopic(self):
        value = (26.5 + 21.13) / 2
        # goal_p = list(self.p_density[0]).index(value)
        goal_p = 1024.5
        isopic = list()

        dens_orig = self.show_potential_density()
        dens_arr = np.zeros((120961, len(self.height_data)))
        for i in range(0, 120961, 1):
            for j in range(0, len(self.height_data)):
                dens_arr[i, j] = round(dens_orig[j, i], ndigits=3)

        for i in range(0, 120961, 1):
            h_indices = [x for x in dens_arr[i] if abs(x - goal_p) < 0.5]
            if len(h_indices) == 0:
                print('нет подходящих точек, нужно поменять интервал')
            h = h_indices.index(min(h_indices, key=lambda x: abs(x-goal_p)))
            isopic.append(self.height_data[h])

        plt.clf()
        plt.plot(np.linspace(0, 14, 120961), -np.array(isopic).transpose(), '-')
        plt.gca().invert_yaxis()
        plt.title("Профиль изопикны")
        plt.xlabel('t, дни')
        plt.ylabel('z, м')
        # plt.show()
        return isopic

    @lru_cache()
    def find_waves(self, isopic):
        maxes = list()
        mins = list()
        t_mins = list()
        is_min = False
        for i in range(1, len(isopic) - 1):
            if isopic[i] < isopic[i-1] and isopic[i] < isopic[i+1] and not is_min:
                mins.append(isopic[i])
                t_mins.append(i)
                is_min = True
            if isopic[i] > isopic[i - 1] and isopic[i] > isopic[i + 1] and is_min:
                maxes.append(isopic[i])
                is_min = False
        waves = list()
        print(len(maxes), len(mins))
        for i in range(0, len(maxes)):
            waves.append(Wave(mins[i], maxes[i], mins[i+1], t_mins[i+1] - t_mins[i]))
        return waves


def mylog(x, mu = 1, sigma = 1):
    return np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi))


def normalise_data(data):
    freq = [data.count(value) / len(data) for value in set(data)]
    return freq

def pirson(data):
    data = normalise_data(data)
    x = 0
    i = 0
    h = 1

    while x <= max(data):
        if data[i] <= (x + h):
            pass
        i += 1



point = SeaPoint('/home/anna/PyProjects/diploma/data/S04_10sec.mat')
# point.show_temp()
# point.show_salinity_profile()
# point.show_potential_density()
isopic = point.find_isopic()

waves = point.find_waves(isopic=tuple(isopic))

heigths = [wave.height() for wave in waves if wave.height() > 0]
height_mean = stat.mean(heigths)
height_deviation = stat.stdev(heigths)
norm_heights = normalise_data(heigths)
dens = mylog(np.array(sorted(list(set(heigths)))), sigma=height_deviation, mu=height_mean)
# print(scipy.stats.chisquare(norm_heights, dens))
print(sum(norm_heights), sum(dens), len(set(heigths)), len(dens))
# pirson(norm_heights, dens)
data = heigths
print([(value, data.count(value)) for value in set(data)])
# гистограмма высоты
plt.clf()
plt.hist(sorted(set(heigths)), edgecolor='black', weights=norm_heights, bins=len(set(heigths)))
plt.plot(sorted(list(set(heigths))), dens, linewidth=2, color='r')
plt.show()

periods = [wave.T for wave in waves]
period_mean = stat.mean(periods)
period_deviation = stat.stdev(periods)
norm_periods = normalise_data(periods)
dens = mylog(np.array(sorted(list(set(periods)))), sigma=period_deviation, mu=period_mean)
print(sum(norm_periods), sum(dens), len(set(periods)), len(dens))
# pirson(norm_periods, dens)
#print(scipy.stats.chisquare(norm_periods, dens))

# гистограмма периода
plt.hist(sorted(set(periods)), edgecolor='black', weights=norm_periods, bins=int(len(set(periods)) / 4))
plt.plot(sorted(list(set(periods))), dens, linewidth=2, color='r')
plt.show()

print(scipy.stats.shapiro(np.array(heigths)))
print(scipy.stats.shapiro(np.array(periods)))

