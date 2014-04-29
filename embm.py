#! /usr/bin/env python3
# 2014-04-11
# Copyright 2014 S. Brewster Malevich <malevich@email.arizona.edu>
    
# Energy-moisture balance climate model based on Fanning and Weaver 1996 (F&W).
# This is a term project for Earth System Modeling (GEOS573) at the 
# University of Arizona.

import numpy as np

SECONDS_PER_YEAR = 3.15569e7

def get_diffusion_coef(lat, coef_type):
    """Calculate the diffusion coefficient for given latitude.

    Args:
        lat: The latitude (in degrees).
        coef_type: Either string "heat" or "moisture" determining which 
            diffusion coefficient is returned.

    Returns:
        Heat (ν) or moisture (κ) diffusion coefficient (m^2/s).
    """
    rad = lat * np.pi/180
    sin_lat = np.sin(rad)
    if coef_type == "heat":
        return 3e6 * (0.81 - 1.08 * sin_lat**2 + 0.74 * sin_lat**4)
    elif coef_type == "moisture":
        abs_sin = np.abs(sin_lat)
        return 1.7e6 * (1.9823 
                        - 17.3501 * abs_sin 
                        + 117.2489 * abs_sin**2 
                        - 274.1129 * abs_sin**3 
                        + 258.2244 * abs_sin**4 
                        - 85.7967 * abs_sin**5)
    else:
        print("'coef_type' argument needs to be either 'heat' or 'moisture'.")
        raise


def get_emissivity(lat, coef_type):
    """Calculate emissivity (ϵ) for given latitude.

    Args:
        lat: The latitude (in degrees).
        coef_type: Either string "atmosphere" or "planet" determining which 
            type of emissivity is returned.

    Returns:
        Heat (ν) or moisture (κ) diffusion coefficient (m^2/s).
    """
    rad = lat * np.pi/180
    sin_lat = np.sin(rad)
    if coef_type == "atmosphere":
        return (0.8666 
                + 0.0408 * sin_lat - 0.2553 * sin_lat**2 
                - 0.466 * sin_lat**3 + 0.9877 * sin_lat**4 
                + 2.0257 * sin_lat**5 - 2.3374 * sin_lat**6
                - 3.199 * sin_lat**7 + 2.8581 * sin_lat**8
                + 1.6070 * sin_lat**9 - 1.2685 * sin_lat**10) 
    elif coef_type == "planet":
        return (0.5531
                - 0.1296 * sin_lat + 0.6796 * sin_lat**2
                + 0.7116 * sin_lat**3 - 2.794 * sin_lat**4
                - 1.3592 * sin_lat**5 + 3.8831 * sin_lat**6
                + 0.8348 * sin_lat**7 - 1.9536 * sin_lat**8)
    else:
        print("'coef_type' argument needs to be either 'atmosphere' or 'planet'.")
        raise


def get_annual_shortwave(lat):
    """Get the annual distribution of shortwave radiation (S) given latitude.
    """
    return 1.5 * (1 - np.sin(lat * np.pi/180)**2)


def get_coalbedo(lat):
    """Get the co-albedo (1 - α) for a given latitude.
    """
    return 0.7995 - 0.315 * np.sin(lat * np.pi/180)**2


def get_vapor_pressure(temp):
    """Get saturated vapor pressure (e_s; mb) for a given temp (K).
    """
    temp_c = temp - 273.15
    return 6.112 * np.exp((17.67 * temp_c)/(temp_c + 243.5))


def get_specific_humidity(temp):
    """Get saturated specific humidity (q_s; g/kg) for a given temp (K).
    """
    vapor_p = get_vapor_pressure(temp)
    return 0.622 * (vapor_p/(1013.26 - 0.378 * vapor_p))


def div(x):
    """Divergence of n-D array x."""
    return np.sum(np.gradient(x), axis = 0)


def weighted_div(x, w):
    """Divergence of 2D array `x`, weighting it with 1D `w` before summing."""
    return np.sum([w[:, np.newaxis] * i for i in np.gradient(x)], axis = 0)


class Embm(object):
    def __init__(self):
        self.n_lon = 72
        self.n_lat = 46
        self.time_step = 3600  # Model time step (s).
        self.earth_radius = 6371  # (km).
        self.rho_air = 1.25  # Air density (kg/m^3).
        self.rho_sea = 1024  # Sea surface density (kg/m^3).
        self.epsilon_sea = 0.65  # Solar scattering coefficient over ocean, c_0.
        self.epsilon_land = 0.3  # Solar scattering coefficient over land, c_0.
        self.c_rhoa = 1e3  # Heat capacity of dry air (j/(kg k)).
        self.emissivity_ocean = 0.96
        self.scale_depth_atmosphere = 8400  # (m).
        self.scale_depth_humidity = 1800  # Specific humidity scale depth (m).
        self.latent_heat_evap = 2.5e6  # Latent heat of evaporation (j/kg).
        self.solar_constant = 1360  # (w/m^2).
        self.stefanboltz = 5.67e-8  # Stefan-Boltzmann constant (w/(m^2 k^4)).

    def setup(self):
        """Initialize the model."""
        self.t = np.ones((self.n_lat, self.n_lon)) * 273.15
        self.q = np.zeros((self.n_lat, self.n_lon))

        # TODO: A lot of this should be spec in a method or initialization.
        self.lat_range = np.linspace(-90, 90, self.n_lat, endpoint = True)
        self.lon_range = np.linspace(-180, 180, self.n_lon, endpoint = True)

        # TODO: This IO should use a method and be done in main().
        self.ocean_mask = np.loadtxt("./data/mask.txt", dtype = "i")
        self.wind = np.loadtxt("./data/wind.txt", dtype = "d")
        self.sst = np.loadtxt("./data/sst.txt", dtype = "f")
        self.sst[self.sst == -999] = np.nan

        # TODO: Need test that all spatial arrays are equal.

        self.diffusion_coef_heat = get_diffusion_coef(self.lat_range, "heat")
        self.diffusion_coef_moisture = get_diffusion_coef(self.lat_range, "moisture")
        self.annual_shortwave = get_annual_shortwave(self.lat_range)
        self.coalbedo = get_coalbedo(self.lat_range)
        self.scattering = np.zeros((self.n_lat, self.n_lon))
        self.scattering[self.ocean_mask == 1] = self.epsilon_sea
        self.scattering[self.ocean_mask == 0] = self.epsilon_land
        self.emissivity_planet = get_emissivity(self.lat_range, "planet")
        self.emissivity_atmosphere = get_emissivity(self.lat_range, "atmosphere")

    def have_rain(self):
        """Return 1 if precipitation, 0 if not at each grid cell"""
        rel_humidity = self.q/get_specific_humidity(self.t)
        out = np.zeros(rel_humidity.shape)
        out[rel_humidity >= 0.85] = 1
        return out

    def update_fluxes(self):
        """Update the ocean and land flux components"""
        # eq. 2a and 2b of F&W.
        self.flux = np.zeros(self.ocean_mask.shape)
        ocean_flux = self.q_t + self.q_ssw - self.q_lw + self.q_rr + self.q_sh + self.q_lh
        land_flux = self.q_t + self.q_ssw - self.q_lw + self.q_lh
        self.flux[self.ocean_mask == 1] = ocean_flux[self.ocean_mask == 1]
        self.flux[self.ocean_mask == 0] = land_flux[self.ocean_mask == 0]

    def step(self, nstep=1):
        """Push the model through `nstep` time steps."""
        self.dalton = 1e-3 * (1.0022 - 0.0822 * (self.t - self.sst) + 0.0266 * self.wind)
        self.stanton = 0.94 * self.dalton
        # Forcing terms
        self.q_ssw = self.solar_constant/4 * self.annual_shortwave[:, np.newaxis] * self.coalbedo[:, np.newaxis] * (1 - self.scattering)  # Q_SSW
        self.q_lw = self.emissivity_planet[:, np.newaxis] * self.stefanboltz * self.t**4  # Q_LW
        self.q_rr = self.emissivity_ocean * self.stefanboltz * self.sst**4 - self.emissivity_atmosphere[:, np.newaxis]* self.stefanboltz * self.t**4  # Q_RR
        self.q_sh = self.rho_air * self.stanton * self.c_rhoa * self.wind * (self.sst - self.t)  # Q_SH
        # Diffusion terms
        self.q_t = self.rho_air * self.scale_depth_atmosphere * self.c_rhoa * weighted_div(self.t, self.diffusion_coef_heat)  # Q_T
        self.m_t = self.rho_air * self.scale_depth_humidity * weighted_div(self.q, self.diffusion_coef_moisture)  # M_T
        # Wet terms
        self.e = (self.rho_air * self.dalton * self.wind * SECONDS_PER_YEAR)/self.rho_sea * (get_specific_humidity(self.sst + 273.15) - self.q)  # E
        self.p = (self.rho_air * self.scale_depth_humidity * SECONDS_PER_YEAR)/(self.rho_sea * self.time_step) * self.have_rain() * (self.q - 0.85 * get_specific_humidity(self.t))  # P
        self.q_lh = (self.rho_sea/SECONDS_PER_YEAR) * self.latent_heat_evap * self.p  # Q_LH
        # Now partial derivs.
        self.update_fluxes()
        self.partial_t = self.flux / (self.rho_air * self.scale_depth_atmosphere * self.c_rhoa)  # partial T/partial t
        self.partial_q = (self.m_t + self.rho_sea * (self.e - self.p)/SECONDS_PER_YEAR)/(self.rho_air * SECONDS_PER_YEAR)  # partial q/partial t
        self.t = self.time_step * self.partial_t
        self.q = self.time_step * self.partial_q
        
        # SECOND UPDATE
        self.dalton = 1e-3 * (1.0022 - 0.0822 * (self.t - self.sst) + 0.0266 * self.wind)
        self.stanton = 0.94 * self.dalton
        # Forcing terms
        self.q_ssw = self.solar_constant/4 * self.annual_shortwave[:, np.newaxis] * self.coalbedo[:, np.newaxis] * (1 - self.scattering)  # Q_SSW
        self.q_lw = self.emissivity_planet[:, np.newaxis] * self.stefanboltz * self.t**4  # Q_LW
        self.q_rr = self.emissivity_ocean * self.stefanboltz * self.sst**4 - self.emissivity_atmosphere[:, np.newaxis]* self.stefanboltz * self.t**4  # Q_RR
        self.q_sh = self.rho_air * self.stanton * self.c_rhoa * self.wind * (self.sst - self.t)  # Q_SH
        # Diffusion terms
        self.q_t = self.rho_air * self.scale_depth_atmosphere * self.c_rhoa * weighted_div(self.t, self.diffusion_coef_heat)  # Q_T
        self.m_t = self.rho_air * self.scale_depth_humidity * weighted_div(self.q, self.diffusion_coef_moisture)  # M_T
        # Wet terms
        self.e = (self.rho_air * self.dalton * self.wind * SECONDS_PER_YEAR)/self.rho_sea * (get_specific_humidity(self.sst + 273.15) - self.q)  # E
        self.p = (self.rho_air * self.scale_depth_humidity * SECONDS_PER_YEAR)/(self.rho_sea * self.time_step) * self.have_rain() * (self.q - 0.85 * get_specific_humidity(self.t))  # P
        self.q_lh = (self.rho_sea/SECONDS_PER_YEAR) * self.latent_heat_evap * self.p  # Q_LH

        # for n in range(nstep):
        #     if n % 10 == 0:
        #         self.q_ssw_new = self.q_ssw
        #     else:
        #         self.t = self.t + 2 * self.time_step * (self.t - self.t)


n_time_step = 500
model = Embm()
model.setup()
# Weighted divergence:
# model.diffusion_coef_heat[:, np.newaxis] * np.gradient(model.t)[1] + model.diffusion_coef_heat[:, np.newaxis] * np.gradient(model.t)[0]

# model.step(n_time_step)


# if __name__ == '__main__':
    # main()