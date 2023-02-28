import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import gsw

class SeaPoint:

    def __init__(self, file):
        # парсим данные из mat-файла и приводим их к нампаевскому виду
        salinity_dict = {"S02": 0, "S04": 33.641, "S05": 0, "S07": 0, "S09": 0}
        latlon_dict = {"S02": 0, "S04": (42, 131), "S05": 0, "S07": 0, "S09": 0}
        mat = sio.loadmat(file)
        temp_header = list(mat.keys())[3]
        height_data_header = list(mat.keys())[8]
        height_header = list(mat.keys())[6]

        self.temp_data = np.array(mat[temp_header])
        self.temp_data = self.temp_data.transpose()
        self.height_data = np.array(mat[height_data_header])[0]
        self.latitude = latlon_dict[temp_header][0]
        self.longitude = latlon_dict[temp_header][1]
        self.height = mat[height_header][0][0]
        print(mat["cS04"])
        self.salinity = salinity_dict[temp_header]


    def show_temp(self):
        fig, ax = plt.subplots(constrained_layout=True)
        y = self.height_data
        dt = np.shape(self.temp_data)[1] #число измерений
        x = np.linspace(0, 14, dt)
        X, Y = np.meshgrid(x, y)
        plt.gca().invert_yaxis()
        Z = self.temp_data
        CS = ax.contourf(X, Y, Z, cmap=plt.cm.CMRmap)
        cbar = fig.colorbar(CS)
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
        self.height_data = -1 * self.height_data #потому что вниз, а не вверх (иначе беды с давлением)
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

point4 = SeaPoint('S04_10sec.mat')
print(point4.salinity)
point4.show_temp()
point4.show_potential_density()

