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
Energy-moisture balance climate model based on [1]_.

Notes
-----
Special thanks to 
Prof. Jianjun Yin at the University of Arizona, Dept. Geosciences for 
input.

References
----------
.. [1] Fanning, A. F., and A. J. Weaver (1996), An atmospheric 
   energy-moisture balance model: Climatology, interpentadal 
   climate change, and coupling to an ocean general 
   circulation model, J. Geophys. Res., 101(D10), 15111–15128, 
   doi:10.1029/96JD01017.
"""


import numpy as np


__all__ = ["vapor_pressure", "specific_humidity", "embm"]


SECONDS_PER_YEAR = 3.15569e7


def vapor_pressure(temp):
    """Return saturated vapor pressure (mb) for a given temp (K).

    Notes
    -----
    Equations for saturated vapor pressure at a given air temp are not 
    give in [1]_. We're using the below definition.

    References
    ----------
    .. [1] Fanning, A. F., and A. J. Weaver (1996), An atmospheric 
       energy-moisture balance model: Climatology, interpentadal 
       climate change, and coupling to an ocean general 
       circulation model, J. Geophys. Res., 101(D10), 15111–15128, 
       doi:10.1029/96JD01017.
    """
    temp_c = temp - 273.15
    return 6.112 * np.exp((17.67*temp_c) / (temp_c+243.5))


def specific_humidity(temp):
    """Return saturated specific humidity (kg/kg) for a given temp (K).

    Notes
    -----
    Equations for saturated specific humidity at a given air temp are 
    not give in [1]_. We're using the below definition.

    References
    ----------
    .. [1] Fanning, A. F., and A. J. Weaver (1996), An atmospheric 
       energy-moisture balance model: Climatology, interpentadal 
       climate change, and coupling to an ocean general 
       circulation model, J. Geophys. Res., 101(D10), 15111–15128, 
       doi:10.1029/96JD01017.
    """
    vapor_p = vapor_pressure(temp)
    return 0.622 * (vapor_p / (1013.26 - 0.378*vapor_p))


class Model(object):
    """Create an EMBM.

    Energy-moisture balance model (EMBM) based on [1]_.

    Attributes
    ----------
    annual_shortwave : (J) array
        Annual shortwave insolation distribution by latitude.
    c_rhoa : float
        Heat capacity of dry air (10^3 * J/(kg K)).
    coalbedo : (J) array
        Also known as (1 - α).
    dalton : (J, I) array
        The Dalton number.
    e : (J, I) array
        Global evaporation (m/yr).
    earth_radius : float
        The radius of the model's planet (m).
    emissivity_atmosphere : (J) array
        Atmospheric emissivity.
    emissivity_ocean : float
        Oceanic emissivity.
    emissivity_planet : (J) array
        Planetary emissivity.
    epsilon_land : float
        Solar scattering coefficient over land.
    epsilon_sea : float
        Solar scattering coefficient over the ocean.
    h_atmosphere : int or float
        Atmospheric scale depth (m).
    h_humid : int or float
        Specific humidity scale depth (m).
    lat_range : (J) array
        Gives the latitude for each of the model's grid cells.
    latent_heat_evap : float
        Latent heat of evaporation (2.5e6 * J/kg).
    lon_range : (I) array
        Gives the latitude for each of the model's grid cells.
    m_t : (J, I) array
        Eddy-diffusive horizontal moisture transport parameterization. 
    nlat : int
        Number of latitudinal cells in the model, J.
    nlon : int
        Number of longitudinal cells in the model, I.
    nu_heat : (J) array
        Eddy diffusivity for heat.
    nu_moisture : (J) array
        Eddy diffusivity for moisture.
    ocean_mask : (J, I) array
        Binary array indicating which of the model's cells are over 
        oceans (`1`) or land (`0`).
    p : (J, I) array
        Precipitation for each of the model's cells (m/yr).
    pcip_flag : (J, I) array
        Binary array indicating whether precipitation is to occur (1) 
        in a given model cell.
    q : (3, J, I) array
        Specific humidity (kg/kg) for a given model cell for three 
        time steps n-1 (0, J, I), n (1, J, I), and n+1 (2, J, I).
    q_lh : (J, I) array
        Latent heat flux into the atmosphere.
    q_lw : (J, I) array
        Infrared emission flux.
    q_rr : (J, I) array
        Radiative flux into the atmosphere.
    q_sh : (J, I) array
        Sensible heat flux from the ocean.
    q_ssw : (J, I) array
        Shortwave radiation absorption flux.
    q_t : (J, I) array
        Eddy-diffusive horizontal heat transport parameterization.
    rho_air : int or float
        Surface air density (kg/m^3).
    rho_sea : int or float
        Sea surface density (kg/m^3).
    solar_constant : int or float
        The solar constant (W/m^3).
    sst : (J, I) array
        Sea-surface temperatures.
    stanton : (J, I) array
        The Stanton number.
    stefanboltz : float
        The Stefan-Boltzmann constant (W/(m^2 * K^4)).
    steps_run : int
        The number of steps the model has run through.
    t : (3, J, I) array
        Air temperature (K) for a given model cell for three 
        time steps n-1 (0, J, I), n (1, J, I), and n+1 (2, J, I).
    time_step : float
        The number of seconds that pass with a single model step.
    wind : (J, I) array
        Wind speed climatology (m/s).
    x_step : (J) array
        The distance (m) covered with each cell in the longitudinal 
        direction.
    y_step : float
        The distance (m) covered with each cell in the latitudinal 
        direction.

    Methods
    -------
    evaluate_evap()
        Evaluate model forcing terms at time n.
    evaluate_forcing()
        Evaluate model forcing terms at time n.
    evaluate_pcip()
        Evaluate the model precipitation variable at time n+1.
    evaluate_q_diffusion()
        Evaluate the model moisture diffusion at time n+1.
    evaluate_t_diffusion()
        Evaluate the model heat diffusion at time n+1.
    global_mean(x)
        Get the grid-area averaged mean for a model variable.
    reset()
        Reset the model's variables to initial state.
    step(nstep=1, trace=False, euler_steps=10)
        Run the model for a period of number of steps.
    step_q_diffusion()
        Update specific humidity at time n+1 from diffusion terms.
    step_t_diffusion()
        Update air temperature at time n+1 from diffusion terms.
    step_t_forcing(euler=False)
        Update air temperature at time n+1 from forcing terms.

    References
    ----------
    .. [1] Fanning, A. F., and A. J. Weaver (1996), An atmospheric 
       energy-moisture balance model: Climatology, interpentadal 
       climate change, and coupling to an ocean general circulation 
       model, J. Geophys. Res., 101(D10), 15111–15128, 
       doi:10.1029/96JD01017.

    Examples
    --------
    >>> import embm
    >>> m = embm.Model()
    >>> m.step(5000)
    >>> m.t[1]  # Air temperature for plotting, etc..
    """

    def __init__(self):
        self._initialize_constants()
        self._initialize_variables()

    def _initialize_constants(self):
        """

        Notes
        -----
        This model breaks the scattering coefficient (C_0) from [1]_ 
        into a coefficient for land and for sea.

        References
        ----------
        .. [1] Fanning, A. F., and A. J. Weaver (1996), An atmospheric 
           energy-moisture balance model: Climatology, interpentadal 
           climate change, and coupling to an ocean general 
           circulation model, J. Geophys. Res., 101(D10), 15111–15128, 
           doi:10.1029/96JD01017.
        """
        self.nlon = 72
        self.nlat = 46
        self.time_step = 3600 / 2
        self.earth_radius = 6371 * 1e3
        self.rho_air = 1.25
        self.rho_sea = 1024
        self.epsilon_sea = 0.65
        self.epsilon_land = 0.3
        self.c_rhoa = 1e3
        self.emissivity_ocean = 0.96
        self.h_atmosphere = 8400
        self.h_humid = 1800
        self.latent_heat_evap = 2.5e6
        self.solar_constant = 1360
        self.stefanboltz = 5.67e-8
        self.lat_range = np.linspace(-90, 90, self.nlat, endpoint = True)
        self.lon_range = np.linspace(-180, 180, self.nlon, endpoint = True)
        self.x_step = (self.earth_radius * 2 
            * np.cos(self.lat_range * np.pi/180) * np.pi / self.nlon)
        self.x_step[0] = 1; self.x_step[-1] = 1
        self.y_step = self.earth_radius * 2 * np.pi / self.nlat
        
        # TODO: This IO should use a method and be done in main().
        self.ocean_mask = np.loadtxt("./data/mask.txt", dtype = "i")
        self.wind = np.loadtxt("./data/wind.txt", dtype = "d")
        self.sst = np.loadtxt("./data/sst.txt", dtype = "f")
        self.sst[self.sst == -999] = np.nan
        self.sst += 273.15  # Convert C to Kelvin.

    def _initialize_variables(self):
        self.steps_run = 0
        self.t = np.ones((3, self.nlat, self.nlon)) * 273.15
        self.q = np.zeros((3, self.nlat, self.nlon))
        self._calc_diffusion_coefs()
        self._calc_annual_shortwave()
        self._calc_coalbedo()
        self._scattering = np.zeros((self.nlat, self.nlon))
        self._scattering[self.ocean_mask == 1] = self.epsilon_sea
        self._scattering[self.ocean_mask == 0] = self.epsilon_land
        self._calc_emissivity()
        self._calc_pcip_flag
        self.p = np.zeros(self.wind.shape)

    def _calc_diffusion_coefs(self):
        """Set the eddy-diffusive coefficients for model latitudes.

        Notes
        -----
        Heat diffusion and moisture diffusion parameterizations 
        (ν and κ, in [1]_, respectively) are plotted but the equation 
        is not given. We're using the below definition.

        References
        ----------
        .. [1] Fanning, A. F., and A. J. Weaver (1996), An atmospheric 
           energy-moisture balance model: Climatology, interpentadal 
           climate change, and coupling to an ocean general 
           circulation model, J. Geophys. Res., 101(D10), 15111–15128, 
           doi:10.1029/96JD01017.
        """
        rad = self.lat_range * np.pi / 180
        sin_lat = np.sin(rad)
        self.nu_heat = 3e6 * (0.81 - 1.08 * sin_lat**2
                                               + 0.74 * sin_lat**4)
        abs_sin = np.abs(sin_lat)
        self.nu_moisture = 1.7e6 * (1.9823
                            - 17.3501 * abs_sin 
                            + 117.2489 * abs_sin**2 
                            - 274.1129 * abs_sin**3 
                            + 258.2244 * abs_sin**4 
                            - 85.7967 * abs_sin**5)

    def _calc_emissivity(self):
        """Set emissivities for model latitudes.

        Notes
        -----
        Atmospheric and planetary emissivity are plotted in [1]_, but 
        no equation is given. We're using the below definition.

        References
        ----------
        .. [1] Fanning, A. F., and A. J. Weaver (1996), An atmospheric 
           energy-moisture balance model: Climatology, interpentadal 
           climate change, and coupling to an ocean general 
           circulation model, J. Geophys. Res., 101(D10), 15111–15128, 
           doi:10.1029/96JD01017.
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
        """Set the annual shortwave radiation for model latitudes.

        Notes
        -----
        The annual shortwave distribution is plotted in [1]_, but no 
        equation is given. We're defining it here as 
        S(φ) = 1.5*(1 - sin^2(φ)).

        References
        ----------
        .. [1] Fanning, A. F., and A. J. Weaver (1996), An atmospheric 
           energy-moisture balance model: Climatology, interpentadal 
           climate change, and coupling to an ocean general 
           circulation model, J. Geophys. Res., 101(D10), 15111–15128, 
           doi:10.1029/96JD01017.
        """
        self.annual_shortwave = 1.5*(1 - np.sin(self.lat_range * np.pi/180)**2)

    def _calc_coalbedo(self):
        """Set the co-albedo for model latitudes.

        Notes
        -----
        The coalbedo (1 - α) is plotted in [1]_, but no 
        equation is given. Defined here as
        (1 - α) = 0.7995 - 0.315*sin^2(φ).

        References
        ----------
        .. [1] Fanning, A. F., and A. J. Weaver (1996), An atmospheric 
           energy-moisture balance model: Climatology, interpentadal 
           climate change, and coupling to an ocean general 
           circulation model, J. Geophys. Res., 101(D10), 15111–15128, 
           doi:10.1029/96JD01017.
        """
        self.coalbedo = 0.7995 - 0.315 * np.sin(self.lat_range * np.pi/180)**2
    
    def _calc_diffusion(self, x, coef):
        """Calculate weighted diffusion.

        Parameters
        ----------
        x : array-like
            Either `self.t` or `self.q`. Needs shape 
            (3, self.lat, self.lon).
        coef : int or float
            Weight coefficient for the gradient of `x`.

        Returns
        -------
        array_like
            Weighted-diffusion array shaped as (self.lat, self.lon).

        Notes
        -----
        This is basically a lazy and cheaply vectorized approach to:
        ∇ • (coef ∇ x)
        """
        # 1st derivative.
        grad1 = np.gradient(x, self.y_step, self.x_step[:, np.newaxis])
        # x component.
        grad1[1] *= coef[:, np.newaxis]
        grad1[1][:, 0] = (x[:, 1] - x[:, -1]) / (2*self.x_step)*coef
        grad1[1][:, -1] = (x[:, 0] - x[:, -2]) / (2*self.x_step)*coef

        # 2nd derivative.
        grad2 = [np.gradient(grad1[0], self.y_step, self.x_step[:, np.newaxis])[0],
                 np.gradient(grad1[1], self.y_step, self.x_step[:, np.newaxis])[1]]
        # y component.
        grad2[0] *= coef[:, np.newaxis]
        # x component.
        grad2[1][:, 0] = (grad1[1][:, 1] - grad1[1][:, -1]) / (2*self.x_step)
        grad2[1][:, -1] = (grad1[1][:, 0] - grad1[1][:, -2]) / (2*self.x_step)
        return grad2[0] + grad2[1]

    def _calc_pcip_flag(self):
        """Set precipitation flags.

        Notes
        -----
        This is from eq. 13 and p113 of [1]_.

        References
        ----------
        .. [1] Fanning, A. F., and A. J. Weaver (1996), An atmospheric 
           energy-moisture balance model: Climatology, interpentadal 
           climate change, and coupling to an ocean general 
           circulation model, J. Geophys. Res., 101(D10), 15111–15128, 
           doi:10.1029/96JD01017.
        """
        rel_humidity = self.q[2] / specific_humidity(self.t[2])
        out = np.zeros(rel_humidity.shape)
        out[rel_humidity >= 0.85] = 1
        self.pcip_flag = out

    def reset(self):
        """Reset the model's variables to initial state.
        """
        self._initialize_variables()

    def evaluate_forcing(self):
        """Evaluate model forcing terms at time n.

        Notes
        -----
        The Dalton number is from eq. 18 of [1]_.

        `q_ssw` is from eq. 4 of [1]_.

        `q_lw` is from eq. 5a of [1]_.

        `q_rr` is from eq. 6 of [1]_.

        `q_sh` is from eq. 7 of [1]_.

        `q_lh` is from eq. 8 of [1]_.

        References
        ----------
        .. [1] Fanning, A. F., and A. J. Weaver (1996), An atmospheric 
           energy-moisture balance model: Climatology, interpentadal 
           climate change, and coupling to an ocean general 
           circulation model, J. Geophys. Res., 101(D10), 15111–15128, 
           doi:10.1029/96JD01017.
        """
        self.dalton = (1e-3 * (1.0022 - 0.0822 * (self.t[1]-self.sst) 
            + 0.0266*self.wind))
        self.stanton = 0.94 * self.dalton
        self.q_ssw = (self.solar_constant/4 * self.annual_shortwave[:, np.newaxis] 
            * self.coalbedo[:, np.newaxis] * (1 - self._scattering))
        self.q_lw = (self.emissivity_planet[:, np.newaxis] 
            * self.stefanboltz * self.t[1]**4)
        self.q_rr = (self.emissivity_ocean * self.stefanboltz * self.sst**4 
            - self.emissivity_atmosphere[:, np.newaxis]*self.stefanboltz*self.t[1]**4)
        self.q_rr[self.ocean_mask == 0] = 0
        self.q_sh = (self.rho_air * self.stanton * self.c_rhoa * self.wind 
            * (self.sst - self.t[1]))
        self.q_sh[self.ocean_mask == 0] = 0
        self.q_lh = ((self.rho_sea/SECONDS_PER_YEAR) 
            * self.latent_heat_evap * self.p)

    def evaluate_evap(self):
        """Evaluate model evaporation variable at time n.

        Notes
        -----
        This is from eq. 11 of [1]_.

        References
        ----------
        .. [1] Fanning, A. F., and A. J. Weaver (1996), An atmospheric 
           energy-moisture balance model: Climatology, interpentadal 
           climate change, and coupling to an ocean general 
           circulation model, J. Geophys. Res., 101(D10), 15111–15128, 
           doi:10.1029/96JD01017.
        """
        self.e = ((self.rho_air * self.dalton * self.wind * SECONDS_PER_YEAR)
            /self.rho_sea * (specific_humidity(self.sst) - self.q[1]))
        self.e[self.ocean_mask == 0] = 0

    def evaluate_pcip(self):
        """Evaluate the model precipitation variable at time n+1.

        Notes
        -----
        This is from eq. 12 of [1]_.

        References
        ----------
        .. [1] Fanning, A. F., and A. J. Weaver (1996), An atmospheric 
           energy-moisture balance model: Climatology, interpentadal 
           climate change, and coupling to an ocean general 
           circulation model, J. Geophys. Res., 101(D10), 15111–15128, 
           doi:10.1029/96JD01017.

        """
        self._calc_pcip_flag()
        self.p = ((self.rho_air * self.h_humid * SECONDS_PER_YEAR)
            /(self.rho_sea*self.time_step) * self.pcip_flag 
            * (self.q[2] - 0.85*specific_humidity(self.t[2])))
        self.q[2][self.pcip_flag == 1] = (0.85 
            * specific_humidity(self.t[2][self.pcip_flag == 1]))

    def evaluate_t_diffusion(self):
        """Evaluate the model heat diffusion at time n+1.

        Notes
        -----
        Defined in eq. 3 of [1]_.

        References
        ----------
        .. [1] Fanning, A. F., and A. J. Weaver (1996), An atmospheric 
           energy-moisture balance model: Climatology, interpentadal 
           climate change, and coupling to an ocean general 
           circulation model, J. Geophys. Res., 101(D10), 15111–15128, 
           doi:10.1029/96JD01017.
        """
        self.q_t = (self.rho_air * self.h_atmosphere * self.c_rhoa 
            * self._calc_diffusion(self.t[2], self.nu_heat))

    def evaluate_q_diffusion(self):
        """Evaluate the model moisture diffusion at time n+1.

        Notes
        -----
        Defined in eq. 10 of [1]_.

        References
        ----------
        .. [1] Fanning, A. F., and A. J. Weaver (1996), An atmospheric 
           energy-moisture balance model: Climatology, interpentadal 
           climate change, and coupling to an ocean general 
           circulation model, J. Geophys. Res., 101(D10), 15111–15128, 
           doi:10.1029/96JD01017.
        """
        self.m_t = (self.rho_air * self.h_humid 
            * self._calc_diffusion(self.q[2], self.nu_moisture))

    def step_t_forcing(self, euler=False):
        """Update air temperature at time n+1 from forcing terms.

        Parameters
        ----------
        euler : bool
            Indicates whether or not to do time differencing with a 
            Euler forward scheme or a leapfrog scheme.

        Notes
        -----
        This uses a leapfrog/Euler-forward scheme. The majority of 
        work is done by a leapfrog scheme but this should be balanced 
        out by performing a Euler forward scheme-based step every now 
        and then (~10 steps with the model's default settings). See 
        [2]_ for a review of these methods.

        References
        ----------
        .. [1] Cushman-Roisin, B., and J. Beckers (2011), Introduction 
           to Geophysical Fluid Dynamics Physical and Numerical 
           Aspects, 2nd ed. Academic Press.
        """
        if euler:
            # Do Euler forward time differencing.
            self.t[2] = (self.t[1] + self.time_step
                /(self.rho_air * self.h_atmosphere * self.c_rhoa) 
                * (self.q_ssw - self.q_lw + self.q_rr + self.q_sh + self.q_lh))
        else:
            # Do leapfrog time differencing.
            self.t[2] = (self.t[0] + 2*self.time_step
                /(self.rho_air * self.h_atmosphere * self.c_rhoa) 
                * (self.q_ssw - self.q_lw + self.q_rr + self.q_sh + self.q_lh))

    def step_t_diffusion(self):
        """Update air temperature at time n+1 from diffusion terms.

        This uses the Matsuno predictor-corrector scheme. It needs to 
        be run two times. See [1]_ for a review.

        References
        ----------
        .. [1] Cushman-Roisin, B., and J. Beckers (2011), Introduction 
           to Geophysical Fluid Dynamics Physical and Numerical 
           Aspects, 2nd ed. Academic Press.
        """
        self.t[2] += (self.time_step * self.q_t
            / (self.rho_air * self.h_atmosphere * self.c_rhoa))
        self.t[2, 0, :] = self.t[2, 1, :].mean()
        self.t[2, -1, :] = self.t[2, -2, :].mean()

    def step_q_diffusion(self):
        """Update specific humidity at time n+1 from diffusion terms.

        This uses the Matsuno predictor-corrector scheme. It needs to 
        be run two times. See [1]_ for a review.

        References
        ----------
        .. [1] Cushman-Roisin, B., and J. Beckers (2011), Introduction 
           to Geophysical Fluid Dynamics Physical and Numerical 
           Aspects, 2nd ed. Academic Press.
        """
        self.q[2] += (self.time_step 
            * (self.m_t + (self.rho_sea * (self.e - self.p))/SECONDS_PER_YEAR)
            / (self.rho_air * self.h_humid))
        self.q[2, 0, :] = self.q[2, 1, :].mean()
        self.q[2, -1, :] = self.q[2, -2, :].mean()

    def global_mean(self, x):
        """Get the grid-area averaged mean for a model variable.
        """
        w = np.repeat(self.x_step * self.y_step, self.nlon)
        return np.average(x.flat, weights = w)

    def step(self, nstep=1, trace=False, euler_steps=10):
        """Run the model for a period of number of steps.

        Parameters
        ----------
        nstep : int, optional
            The number of time steps to run through. Default is 1.
        trace : bool, optional
            Indicating whether you would like the mean specific 
            humidity and air temperature averages for each time step 
            to be stored and returned. Default is `False`.
        euler_steps : int, optional
            After how many time steps the temperature forcing 
            integration should switch from a Leapfrog scheme to a 
            Euler forward scheme. The default is 10 steps.

        Returns
        -------
        t_hist : array-like
            Only returned if `trace = True`. Array giving the 
            evolution of the model's air temperature as it steps 
            through time.
        q_hist : array-like
            Only returned if `trace = True`. Array giving the 
            evolution of the model's specific humidity as it 
            steps through time.

        Notes
        -----
        We're breaking down the contribution to changes in `t` and `q` 
        into separate time differencing schemes. This is a different 
        approach from that used in [1]_. See [2]_ for a review of 
        these methods.

        References
        ----------
        .. [2] Cushman-Roisin, B., and J. Beckers (2011), Introduction 
           to Geophysical Fluid Dynamics Physical and Numerical 
           Aspects, 2nd ed. Academic Press.
        .. [1] Fanning, A. F., and A. J. Weaver (1996), An atmospheric 
           energy-moisture balance model: Climatology, interpentadal 
           climate change, and coupling to an ocean general 
           circulation model, J. Geophys. Res., 101(D10), 15111–15128, 
           doi:10.1029/96JD01017.
        """
        if trace:
            t_hist = np.zeros(nstep)
            q_hist = np.zeros(nstep)

        for i in range(nstep):
            # Time step
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
                # TODO: Check why we can't assign to v[2]. Something is off here?
                # v[2] =
            if trace:
                t_hist[i] = self.global_mean(self.t[1])
                q_hist[i] = self.global_mean(self.q[1])
            self.steps_run += 1
        if trace:
            return t_hist, q_hist
