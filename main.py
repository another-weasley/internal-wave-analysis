from SeaPoint import SeaPoint, show_salinity_profile
from MyStats import MyStat

def main():
    point = SeaPoint('/home/anna/PyProjects/diploma/data/S09_10sec.mat')
    # point.show_temp()
    # show_salinity_profile()
    # point.show_potential_density()
    isopic = point.show_isopic() # работает при выключенном показе плотности
    waves = point.find_waves(isopic=tuple(isopic))

    stat_manager = MyStat(waves)
    # stat_manager.show_height_hist()
    # stat_manager.show_period_hist()
    # stat_manager.show_slope_hist()
    # stat_manager.show_spectral(isopic)
    stat_manager.show_angle_hist(isopic)


if __name__ == "__main__":
    main()
