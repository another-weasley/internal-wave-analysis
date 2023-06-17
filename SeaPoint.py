import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import gsw
from scipy import interpolate
from scipy.ndimage import gaussian_filter
import pandas
from Wave import Wave

matplotlib.rcParams['font.family'] = 'serif'
plt.rc('font', size=14)
LATITUDE = 42
LONGITUDE = 131

WOA_SALINITY = {5: 33.035, 10: 33.292, 15: 33.363, 20: 33.435, 25: 33.492, 30: 33.532, 35: 33.590, 40: 33.641,
                45: 33.712, 50: 33.761, 55: 33.830, 60: 33.868, 65: 33.902, 70: 33.929, 75: 33.954, 80: 33.983,
                85: 33.995, 90: 34.003, 95: 34.012, 100: 34.018}

FIXED_VALUES = {'S02': 1023.7, 'S04': 1023.4, 'S05': 1023.9, 'S07': 1023.4, 'S09': 1023.7}

MINUTES = 20160


def show_salinity_profile():
    keys = np.array(list(WOA_SALINITY.keys()))
    values = np.array(list(WOA_SALINITY.values()))
    plt.gca().invert_yaxis()
    plt.plot(values, keys)
    plt.title("Профиль солености")
    plt.xlabel('S, %')
    plt.ylabel('z, м')
    plt.show()


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
        self.latitude = LATITUDE
        self.longitude = LONGITUDE
        self.height = mat[height_header][0][0]
        self.measure_count = len(self.temp_data[0])
        self.name = list(mat.keys())[3]

        self.salinity = []
        for h in self.height_data:
            for key, value in WOA_SALINITY.items():
                if abs(h-key) <= 2.5:
                    self.salinity.append(value)
                    break
        self.salinity = np.array(self.salinity)

        print(self.height_data)

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


    def calc_density(self):
        self.height_data = -1 * self.height_data  # потому что вниз, а не вверх (иначе беды с давлением)
        pressure = gsw.p_from_z(self.height_data, self.latitude)
        absolute_salinity = gsw.SA_from_SP(self.salinity, pressure, self.longitude, self.latitude)
        # приведем аболютную соленость к такому же виду, в каком лежит температура (была ошибка)
        abs_sal_arr = np.zeros((len(self.height_data), self.measure_count))
        for i in range(0, len(self.height_data), 1):
            abs_sal_arr[i, :] = absolute_salinity[i]
        density_data = gsw.density.sigma0(abs_sal_arr, self.temp_data) + 1000

        print(self.height_data[9])
        print(density_data[0][9])
        return density_data

    def show_potential_density(self):
        fig, ax = plt.subplots(constrained_layout=True)
        y = self.height_data
        dt = np.shape(self.temp_data)[1]  # число измерений
        x = np.linspace(0, 14, dt)
        X, Y = np.meshgrid(x, y)
        plt.gca().invert_yaxis()
        density_data = self.calc_density()
        cs = ax.contourf(X, Y, density_data, cmap=plt.cm.jet)
        norm = matplotlib.colors.Normalize(vmin=cs.cvalues.min(), vmax=cs.cvalues.max())
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cs.cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ticks=cs.levels)
        cbar.ax.set_ylabel("плотность, кг/м^3")
        ax.set_title("Потенциальная плотность")
        ax.set_xlabel("t, дни")
        ax.set_ylabel("z, м")
        plt.show()

    def calc_isopic(self):
        goal_p = FIXED_VALUES[self.name]
        #goal_p = 1024.5
        fig, ax = plt.subplots(constrained_layout=True)
        y = self.height_data
        dt = np.shape(self.temp_data)[1]  # число измерений
        x = np.linspace(0, MINUTES, dt)
        X, Y = np.meshgrid(x, y)
        density_data = self.calc_density()
        isopic = ax.contour(X, Y, density_data, levels=[goal_p])
        plt.clf()

        allpoint1 = np.array([])
        allpoint2 = np.array([])
        # собираем все контуры в один
        for ii, seg in enumerate(isopic.allsegs[0]):
            allpoint1 = np.hstack([allpoint1, np.array(seg[:, 0]).flatten()])
            allpoint2 = np.hstack([allpoint2, np.array(seg[:, 1]).flatten()])
        # сортируем контур по значениям температуры (чтобы не было полос)
        points = zip(allpoint1, allpoint2)
        points = sorted(points, key=lambda p: p[0])
        allpoint1 = [p[0] for p in points]
        allpoint2 = [p[1] for p in points]
        # удаляем дубликаты из контура
        df = pandas.DataFrame({'x': allpoint1, 'y': allpoint2})
        df = df.drop_duplicates(subset=['x'])
        npdf = df.to_numpy().transpose()
        allpoint1, allpoint2 = npdf[0], npdf[1]
        # интерполируем контур
        x = np.linspace(0, MINUTES, MINUTES * 6)
        isof = interpolate.interp1d(allpoint1, allpoint2, kind='linear')
        isopic = isof(x)
        isopic = gaussian_filter(isopic, sigma=10)

        # изопикна c дискретностью 10с
        return isopic

    def show_isopic(self):
        isopic = self.calc_isopic()
        gauss_iso = gaussian_filter(isopic, sigma=6)
        x = np.linspace(0, MINUTES, MINUTES * 6)
        plt.plot(x, isopic)
        plt.title("Профиль изопикны")
        plt.xlabel('t, минуты')
        plt.ylabel('z, м')
        plt.show()
        return isopic

    def find_waves(self, isopic):
        maxes = list()
        mins = list()
        t_mins = list()
        is_min = False
        for i in range(1, len(isopic) - 1):
            if isopic[i] < isopic[i - 1] and isopic[i] < isopic[i + 1] and not is_min:
                mins.append(isopic[i])
                t_mins.append(i)
                is_min = True
            elif isopic[i] < isopic[i - 1] and isopic[i] < isopic[i + 1] and is_min:
                mins.pop()
                mins.append(isopic[i])
                t_mins.pop()
                t_mins.append(i)
                is_min = True
            if isopic[i] > isopic[i - 1] and isopic[i] > isopic[i + 1] and is_min:
                maxes.append(isopic[i])
                is_min = False
            elif isopic[i] > isopic[i - 1] and isopic[i] > isopic[i + 1] and not is_min:
                try:
                    maxes.pop()
                except Exception:
                    pass
                maxes.append(isopic[i])
                is_min = False
        waves = list()
        for i in range(0, min(len(maxes), len(mins)) - 1):
            waves.append(Wave(mins[i], maxes[i], mins[i + 1], t_mins[i + 1] - t_mins[i]))
        return waves
