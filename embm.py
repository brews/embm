# 2014-04-11
# Copyright 2014 S. Brewster Malevich <malevich@email.arizona.edu>
#
#   embm is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>


"""
Energy-moisture balance climate model based on Fanning and Weaver 1996 (F&W).
"""

import numpy as np
import pylab as plt
from tqdm import tqdm


SECONDS_PER_YEAR = 3.15569e7


def get_vapor_pressure(temp):
    """Get saturated vapor pressure (e_s; mb) for a given temp (K)
    """
    temp_c = temp - 273.15
    return 6.112 * np.exp((17.67 * temp_c)/(temp_c + 243.5))


def get_specific_humidity(temp):
    """Get saturated specific humidity (q_s; kg/kg) for a given temp (K)
    """
    vapor_p = get_vapor_pressure(temp)
    return 0.622 * (vapor_p/(1013.26 - 0.378 * vapor_p))


class Model(object):


    def __init__(self):
        self._initialize_constants()
        self._initialize_variables()


    def _initialize_constants(self):
        self.n_lon = 72
        self.n_lat = 46
        self.time_step = 3600/2  # Model time step (s).
        self.earth_radius = 6371 * 1e3  # (m).
        self.rho_air = 1.25  # Air density (kg/m^3).
        self.rho_sea = 1024  # Sea surface density (kg/m^3).
        self.epsilon_sea = 0.65  # Solar scattering coefficient of ocean, c_0.
        self.epsilon_land = 0.3  # Solar scattering coefficient of land, c_0.
        self.c_rhoa = 1e3  # Heat capacity of dry air (J/(kg K)).
        self.emissivity_ocean = 0.96
        self.scale_depth_atmosphere = 8400  # (m).
        self.scale_depth_humidity = 1800  # Specific humidity scale depth (m).
        self.latent_heat_evap = 2.5e6  # Latent heat of evaporation (J/kg).
        self.solar_constant = 1360  # (w/m^2).
        self.stefanboltz = 5.67e-8  # Stefan-Boltzmann constant (w/(m^2 k^4)).

        # TODO: A lot of this should be spec in a method or initialization.
        self.lat_range = np.linspace(-90, 90, self.n_lat, endpoint = True)
        self.lon_range = np.linspace(-180, 180, self.n_lon, endpoint = True)
        self.x_step = (self.earth_radius * 2 
            * np.cos(self.lat_range * np.pi/180) * np.pi / self.n_lon)  # (m)
        self.x_step[0] = 1; self.x_step[-1] = 1  # Not sure about this.
        self.y_step = self.earth_radius * 2 * np.pi / self.n_lat  # (m)
        
        # TODO: This IO should use a method and be done in main().
        self.ocean_mask = np.loadtxt("./data/mask.txt", dtype = "i")
        self.wind = np.loadtxt("./data/wind.txt", dtype = "d")
        self.sst = np.loadtxt("./data/sst.txt", dtype = "f")
        self.sst[self.sst == -999] = np.nan
        self.sst += 273.15


    def _initialize_variables(self):
        self.steps_run = 0
        self.t = np.ones((3, self.n_lat, self.n_lon)) * 273.15
        self.q = np.zeros((3, self.n_lat, self.n_lon))
        self._calc_diffusion_coefs()
        self._calc_annual_shortwave()
        self._calc_coalbedo()
        self.scattering = np.zeros((self.n_lat, self.n_lon))
        self.scattering[self.ocean_mask == 1] = self.epsilon_sea
        self.scattering[self.ocean_mask == 0] = self.epsilon_land
        self._calc_emissivity()
        self.calc_pcip_flag()
        self.p = np.zeros(self.wind.shape)


    def _calc_diffusion_coefs(self):
        """Calculate the diffusion coefficients for model latitudes

        Set model heat (ν) and moisture (κ) diffusion coefficient (m^2/s).
        """
        rad = self.lat_range * np.pi/180
        sin_lat = np.sin(rad)
        self.diffusion_coef_heat = 3e6 * (0.81 - 1.08 * sin_lat**2 
                                               + 0.74 * sin_lat**4)
        abs_sin = np.abs(sin_lat)
        self.diffusion_coef_moisture = 1.7e6 * (1.9823 
                            - 17.3501 * abs_sin 
                            + 117.2489 * abs_sin**2 
                            - 274.1129 * abs_sin**3 
                            + 258.2244 * abs_sin**4 
                            - 85.7967 * abs_sin**5)


    def _calc_emissivity(self):
        """Calculate emissivity (ϵ) for given latitude

        Sets the model atmosphere and planet emissivity.
        """
        rad = self.lat_range * np.pi/180
        sin_lat = np.sin(rad)
        self.emissivity_atmosphere = (0.8666 
                    + 0.0408 * sin_lat - 0.2553 * sin_lat**2 
                    - 0.466 * sin_lat**3 + 0.9877 * sin_lat**4 
                    + 2.0257 * sin_lat**5 - 2.3374 * sin_lat**6
                    - 3.199 * sin_lat**7 + 2.8581 * sin_lat**8
                    + 1.6070 * sin_lat**9 - 1.2685 * sin_lat**10) 
        self.emissivity_planet = (0.5531
                    - 0.1296 * sin_lat + 0.6796 * sin_lat**2
                    + 0.7116 * sin_lat**3 - 2.794 * sin_lat**4
                    - 1.3592 * sin_lat**5 + 3.8831 * sin_lat**6
                    + 0.8348 * sin_lat**7 - 1.9536 * sin_lat**8)


    def _calc_annual_shortwave(self):
        """Set the annual distribution of shortwave radiation (S) given latitude
        """
        self.annual_shortwave = 1.5 * (1 - np.sin(self.lat_range * np.pi/180)**2)


    def _calc_coalbedo(self):
        """Get the co-albedo (1 - α) for a given latitude"""
        self.coalbedo = 0.7995 - 0.315 * np.sin(self.lat_range * np.pi/180)**2

    
    def _calc_diffusion(self, x, coef):
        # 1st derivative.
        grad1 = np.gradient(x, self.y_step, self.x_step[:, np.newaxis])
        # x component.
        grad1[1] *= coef[:, np.newaxis]
        grad1[1][:, 0] = (x[:, 1] - x[:, -1]) / (2 * self.x_step) * coef
        grad1[1][:, -1] = (x[:, 0] - x[:, -2]) / (2 * self.x_step) * coef

        # 2nd derivative.
        grad2 = [np.gradient(grad1[0], self.y_step, self.x_step[:, np.newaxis])[0],
                 np.gradient(grad1[1], self.y_step, self.x_step[:, np.newaxis])[1]]
        # y component.
        grad2[0] *= coef[:, np.newaxis]
        # x component.
        grad2[1][:, 0] = (grad1[1][:, 1] - grad1[1][:, -1]) / (2 * self.x_step)
        grad2[1][:, -1] = (grad1[1][:, 0] - grad1[1][:, -2]) / (2 * self.x_step)
        return grad2[0] + grad2[1]


    def reset(self):
        """Reset the model's variables"""
        self._initialize_variables()


    def evaluate_forcing(self):
        """Evaluate forcing terms at time `n`"""
        # TODO: Be sure we're accounting for all terms in eq 2. and ocean/land differences.
        self.dalton = (1e-3 * (1.0022 - 0.0822 * (self.t[1] - self.sst) 
            + 0.0266 * self.wind))

        self.stanton = 0.94 * self.dalton

        self.q_ssw = (self.solar_constant/4 * self.annual_shortwave[:, np.newaxis] 
            * self.coalbedo[:, np.newaxis] * (1 - self.scattering))  # Q_SSW

        self.q_lw = (self.emissivity_planet[:, np.newaxis] 
            * self.stefanboltz * self.t[1]**4)

        self.q_rr = (self.emissivity_ocean * self.stefanboltz * self.sst**4 
            - self.emissivity_atmosphere[:, np.newaxis] * self.stefanboltz 
            * self.t[1]**4)  # Q_RR

        self.q_rr[self.ocean_mask == 0] = 0
        self.q_sh = (self.rho_air * self.stanton * self.c_rhoa * self.wind 
            * (self.sst - self.t[1]))

        self.q_sh[self.ocean_mask == 0] = 0
        self.q_lh = ((self.rho_sea/SECONDS_PER_YEAR) 
            * self.latent_heat_evap * self.p)


    def evaluate_evap(self):
        """Evaluate the evaporation terms at `n`"""
        self.e = ((self.rho_air * self.dalton * self.wind * SECONDS_PER_YEAR)
            /self.rho_sea * (get_specific_humidity(self.sst) - self.q[1]))

        self.e[self.ocean_mask == 0] = 0
        # self.q_lh = (self.rho_sea/SECONDS_PER_YEAR) * self.latent_heat_evap * self.p  # Q_LH


    def evaluate_pcip(self):
        """Evaluate the precipitation terms at `n + 1`"""
        self.calc_pcip_flag()
        self.p = ((self.rho_air * self.scale_depth_humidity * SECONDS_PER_YEAR)
            /(self.rho_sea * self.time_step) * self.pcip_flag 
            * (self.q[2] - 0.85 * get_specific_humidity(self.t[2])))

        self.q[2][self.pcip_flag == 1] = (0.85 
            * get_specific_humidity(self.t[2][self.pcip_flag == 1]))


    def evaluate_t_diffusion(self):
        """Evaluate heat diffusion at time `n + 1`"""
        self.q_t = (self.rho_air * self.scale_depth_atmosphere * self.c_rhoa 
            * self._calc_diffusion(self.t[2], self.diffusion_coef_heat))


    def evaluate_q_diffusion(self):
        """Evaluate moisture diffusion at time `n + 1`"""
        self.m_t = (self.rho_air * self.scale_depth_humidity 
            * self._calc_diffusion(self.q[2], self.diffusion_coef_moisture))


    def step_t_forcing(self, euler=False):
        """Update air temperature at `n + 1` based on change in forcing

        This uses a leapfrog/Euler-forward scheme.
        """
        if euler:
            # Do Euler forward time differencing.
            self.t[2] = (self.t[1] + self.time_step
                /(self.rho_air * self.scale_depth_atmosphere * self.c_rhoa) 
                * (self.q_ssw - self.q_lw + self.q_rr + self.q_sh + self.q_lh))

        else:
            # Do leapfrog time differencing.
            self.t[2] = (self.t[0] + 2 * self.time_step
                /(self.rho_air * self.scale_depth_atmosphere * self.c_rhoa) 
                * (self.q_ssw - self.q_lw + self.q_rr + self.q_sh + self.q_lh))


    def step_t_diffusion(self):
        """Update air temperature at `n + 1` based on change in diffusion terms

        This uses the Matsuno predictor-corrector scheme.
        """
        self.t[2] += (self.time_step * self.q_t
            / (self.rho_air * self.scale_depth_atmosphere * self.c_rhoa))

        self.t[2, 0, :] = self.t[2, 1, :].mean()
        self.t[2, -1, :] = self.t[2, -2, :].mean()


    def step_q_diffusion(self):
        """Update specific humidity at `n + 1` based on change in diffusion terms

        This uses the Matsuno predictor-corrector scheme.
        """
        self.q[2] += (self.time_step 
            * (self.m_t + (self.rho_sea * (self.e - self.p))/SECONDS_PER_YEAR)
            / (self.rho_air * self.scale_depth_humidity))

        self.q[2, 0, :] = self.q[2, 1, :].mean()
        self.q[2, -1, :] = self.q[2, -2, :].mean()


    def calc_pcip_flag(self):
        """Return 1 if precipitation, 0 if not at each grid cell"""
        # TODO: Which `n` do we want this at?
        rel_humidity = self.q[2]/get_specific_humidity(self.t[2])
        out = np.zeros(rel_humidity.shape)
        out[rel_humidity >= 0.85] = 1
        self.pcip_flag = out


    def global_mean(self, x):
        """Get the grid-area averaged mean for a model variable.
        """
        return np.average(x.flat, weights = np.repeat(self.x_step * self.y_step, self.n_lon))


    def step(self, nstep=1, trace=False, euler_steps=10, verbose=False):
        """Run the model for a period of time_step

        Args:
            nstep: The number of time steps to run through. Default is 1.
            trace: Either `True` or `False` indicating whether you would like 
                the mean specific humidity and air temperature averages for 
                each time step to be stored and returned. Default is `False`.
            euler_steps: After how many time steps the temperature forcing 
                integration should switch from a Leapfrog scheme to a Euler 
                forward scheme. The default is 10 steps.
            verbose: Either `True` or `False` to showing a progress bar in 
                the console. Handy for long runs. Default is false.

        Returns:
            Nothing unless `euler_steps = True`, then `t_history`, and 
            `q_history` are returned.
        """
        def ranger():
            if verbose:
                return tqdm(range(nstep))
            else:
                return range(nstep)

        if trace:
            t_hist = np.zeros(nstep)
            q_hist = np.zeros(nstep)

        for i in ranger():
            self.evaluate_forcing()
            self.evaluate_evap()

            # Leapfrog/Euler-forward step
            if i % euler_steps:
                self.step_t_forcing(euler = True)
            else:
                self.step_t_forcing()

            # Predictor step
            self.evaluate_t_diffusion()
            self.step_t_diffusion()
            self.evaluate_q_diffusion()
            self.step_q_diffusion()

            # Corrector step
            self.evaluate_t_diffusion()
            self.step_t_diffusion()
            self.evaluate_q_diffusion()
            self.step_q_diffusion()

            self.evaluate_pcip()

            # Shifting one step forward in time.
            for v in [self.t, self.q]:
                v[0] = np.copy(v[1])
                v[1] = np.copy(v[2])
                # TODO: Check why we can't assign to v[2]. Something is off here.
                # v[2] =
            if trace:
                t_hist[i] = self.global_mean(self.t[1])
                q_hist[i] = self.global_mean(self.q[1])
            self.steps_run += 1
        if trace:
            return t_hist, q_hist
