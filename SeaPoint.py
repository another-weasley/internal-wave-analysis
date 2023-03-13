import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import gsw


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
        self.salinity = {5: 33.035, 10: 33.292, 15: 33.363, 20: 33.435, 25: 33.492,
                         30: 33.532, 35: 33.590, 40: 33.641, 45: 33.712, 50: 33.761,
                         55: 33.830, 60: 33.868, 65: 33.902, 70: 33.929, 75: 33.954,
                         80: 33.983, 85: 33.995, 90: 34.003, 95: 34.012, 100: 34.018}

    def show_temp(self):
        fig, ax = plt.subplots(constrained_layout=True)
        y = self.height_data
        dt = np.shape(self.temp_data)[1]  # число измерений
        x = np.linspace(0, 14, dt)
        X, Y = np.meshgrid(x, y)
        plt.gca().invert_yaxis()
        Z = self.temp_data
        cs = ax.contourf(X, Y, Z, cmap=plt.cm.jet)
        norm = matplotlib.colors.Normalize(vmin=cs.cvalues.min(), vmax=cs.cvalues.max())
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cs.cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ticks=cs.levels)
        #cbar = fig.colorbar(cs)
        cbar.ax.set_ylabel("температура, С")
        ax.set_title("Поле температуры")
        ax.set_xlabel("t, дни")
        ax.set_ylabel("h, м")
        plt.show()

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
        abs_sal_arr = np.zeros((36, 120961))
        for i in range(0, 36, 1):
            abs_sal_arr[i, :] = absolute_salinity[i]
        density_data = gsw.density.sigma0(abs_sal_arr, self.temp_data)
        CS = ax.contourf(X, Y, density_data, cmap=plt.cm.CMRmap_r)
        cbar = fig.colorbar(CS)
        cbar.ax.set_ylabel("плотность, кг/м^3")
        ax.set_title("Потенциальная плотность")
        ax.set_xlabel("t, дни")
        ax.set_ylabel("h, м")
        plt.show()


point = SeaPoint('/home/anna/PyProjects/diploma/data/S09_10sec.mat')
point.show_temp()
# point.show_potential_density()
