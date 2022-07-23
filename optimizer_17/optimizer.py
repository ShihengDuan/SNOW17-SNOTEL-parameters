from spotpy.parameter import Uniform
from spotpy.objectivefunctions import rmse
from spotpy.examples.hymod_python.hymod import hymod
import os
from snow17 import snow17
import xarray as xa
import pandas as pd


class spot_setup(object):
    # parameters. First four are the main parameters.
    scf = Uniform(low=0.9, high=1.2)
    mfmin = Uniform(low=0.1, high=0.6)
    mfmax = Uniform(low=0.5, high=1.3)
    UADJ = Uniform(low=0.05, high=0.2)
    pxtemp = Uniform(low=0.0, high=2.0)
    pxtemp1 = Uniform(low=-2.0, high=0.0)
    pxtemp2 = Uniform(low=0.0, high=4.0)

    def __init__(self, n_station, obj_func=None):
        self.obj_func = obj_func
        # Get input data
        data = xa.open_dataset(
            '/tempest/duan0000/snow17/data/raw_wus_snotel_topo_clean.nc')
        '''pr = data.precip.isel(n_stations=n_station).sel(
            time=slice('1999-08-01', '2002-08-01'))*25.4
        tave = data.mean_temperature.isel(n_stations=n_station).sel(
            time=slice('1999-08-01', '2002-08-01'))
        tave = tave.interpolate_na(dim="time", method="linear")
        tave = (tave-32)*5/9'''
        pr = xa.open_dataarray('/tempest/duan0000/snow17/snotel/pr_wus_clean.nc').isel(
            n_stations=n_station).sel(time=slice('1999-08-01', '2011-08-15'))
        tave = xa.open_dataarray('/tempest/duan0000/snow17/snotel/tave_wus_clean.nc').isel(
            n_stations=n_station).sel(time=slice('1999-08-01', '2011-08-15'))
        tave = tave-273
        self.latitude = data.latitude.isel(n_stations=n_station).data
        self.elevation = data.elevation_prism.isel(n_stations=n_station).data
        swe = data.SWE.isel(n_stations=n_station).sel(
            time=slice('1999-08-01', '2011-08-15'))*25.4
        swe = swe.interpolate_na(dim='time', method='linear')
        swe = swe.fillna(0)
        self.real_swe = swe
        time = self.real_swe.time.data
        time_datetime = [pd.to_datetime(t) for t in time]
        self.time = time_datetime
        self.precip = pr.data
        self.temperature = tave.data

    def simulation(self, x):
        # Here the model is actualy started with a unique parameter combination that it gets from spotpy for each time the model is called
        model_swe, outflow = snow17(self.time, self.precip, self.temperature, lat=self.latitude, elevation=self.elevation,
                                    scf=x[0], mfmin=x[1], mfmax=x[2], uadj=x[3], pxtemp=x[4], pxtemp1=x[5], pxtemp2=x[6], rvs=2)

        return model_swe[:]

    def evaluation(self):
        # The first year of simulation data is ignored (warm-up)
        return self.real_swe[:]

    def objectivefunction(self, simulation, evaluation, params=None):
        # SPOTPY expects to get one or multiple values back,
        # that define the performance of the model run
        if not self.obj_func:
            # This is used if not overwritten by user
            like = rmse(evaluation, simulation)
        else:
            # Way to ensure flexible spot setup class
            like = self.obj_func(evaluation, simulation)
        return like
