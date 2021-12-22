
#Import Python Libraries
import numpy as np
import matplotlib.pyplot as plot 
from tqdm import tqdm
import math
from datetime import datetime as DateTime
from datetime import date
import json

#Setting
day_night = 1                     # Parameter to turn on/off day-night cycle; 1: day-night mode; 0: constant illumination
miro_switch = 0                   # Enable MIRO Mode: 0 = OFF, 1 = DAY, 2 = NIGHT (both night cases together)
                                  # import of parameters via tables (Heliocentric Distance, Tilt) and choice of output time
                                  # "tilt" only varies the input intensity, not the length of the day!
                                  # Start of Simulation shiftet to noon for MIRO mode, default is sunrise.
number_of_days = 1000             # for MIRO Mode: number of days before fixed time (day/night) COMET DAY!!! 5683
pebble_case = 1                   # Enable "Pebble Case" (Thermal Conductivity gets Temperature and Pebble Radius Dependency)
elliptical_orbit = 1              # Enable an elliptical orbit
dx_switch = 0                     # Enable dx Variation with Depth: linear == 1, log == 2, else constant min_dx
                                  # if == 0: calculation of n via total depth - for usage of user-n, choose dx_switch > 2
VFF_switch = 0                    # Enable Variation of Volume Filling Factor for upper pebble layers - therefore only for Pebble Case
Fourier_break = 0                 # If == 1: Break of run if Fourier number > 0.5, else: only printed
Output_switch = 0                 # Plots and Tables are saved in folders
depth_dependant_tensile_strength = 0 # Enable depth dependant tensile strength. Else constant tensile strength specified below.
ejection = 1                      # Enable ejection of upper layers due to pressure
save = 1                          # Enable save of the output arrays to load them back again into the code
load = 0                          # Use a save instead of the standard starting values for the arrays (it is assumed the save data is in the same path as the python script)


#Definition of Simulation Parameters
min_dx = 5E-3                     # Layer Thickness of smallest layer, Numerical Resolution  [m]     
dt = 10                           # Timestep                               [s]
n = 500                            # Number of Layers                       [-]
k = 2 * 434000                       # Number of Timesteps                    [-]
total_depth = 2.5                   # Total simulated depth                  [m]
day_counter = 0                   # Store of Number of Days for a Run      [-]

#Definition of Physical Parameters
#Material Properties
lambda_constant = 1E-2            # Thermal Conductivity                   [W/(K m)]
density_nucleus = 532             # Density of the Material                [kg/m^3]
specific_heat_capacity = 560      # Specific Heat Capacity                 [J/(kg K)]
r_mono = 1E-6                     # Radius of Monomeres                    [m]
r_agg = 5E-3                      # Radius of Pebbles                      [m]
e_1 = 1.34                        # Parameter for mean free path           [-]
VFF_pack_const = 0.6              # Volume Filling Factor of Packing       [-]
VFF_agg_base = 0.4                # Volume Filling Factor of Pebble        [-]
poisson_ratio_agg = 0.17          # Poisson`s ratio of Pebble              [-]
poisson_ratio_par = 0.17          # Poisson`s ratio of Particle            [-]
young_modulus_agg = 8.1E3         # Young`s modulus of Peblle              [Pa] 
young_modulus_par = 5.5E10        # Young`s modulus of Particle            [Pa] 
surface_energy_par = 0.014          # specific surface energy of Particle    [J/m^2]
f_1 = 5.18E-2                     # empirical constant for Packing Structure Factor 1 [-]
f_2 = 5.26                        # empirical constant for Packing Structure Factor 2 [-]
dust_ice_ratio_global = 3         # global dust to ice ratio of the comet  [-]
co2_h2o_ratio_global = 0.15       # Percentage of CO2 ice content of total ice content [-]
dust_layer_thickness = 0          # thickness of the dust layer of the comet in pebble radii [-]
m_H2O = 2.99E-26                  # mass of a water molecule               [kg]
m_CO2 = 7.31E-26
b = 1 * (2*r_agg)                          # Permeability Coefficient               [-]
density_water_ice = 934           # Density if water ice at around 90K     [kg/m^3]
density_co2_ice = 1600
molar_mass_water = 18.015E-3    # Molar mass of water                    [kg/mol]
molar_mass_dust = 140.706E-3          # This is a placeholder (molar mass of forsterite, a component of comet dust)
molar_mass_co2 = 44.010E-3
heat_capacity_dust = 3000         # Heat Capacity of dust                  [J/(kg * K)]
heat_capacity_water_ice = 30      # Heat Capacity of water ice             [J/(kg * K)]
heat_capacity_co2_ice = 850       # Heat Capacity of CO2 ice               [J/(kg * K)]
latent_heat_water = 2.86E6        # Latent heat of water ice               [J/kg]
latent_heat_co2 = 0.57E6          # Latent heat of CO2 ice                 [J/kg]
depth_dependant_strength = 1      # Parameter used to calculate the tensile strength [Pa]
const_tensile_strength = 0.05       # Tensile strength of the comet material [Pa]
x_0 = 5E-2                        # Length scaling factor used to calculate the tensile strength [m]
#Thermal Properties
temperature_ini = 50              # Start Temperature                      [K]
Input_Intensity = 100             # Intensity of the Light Source (Sun)    [W/m^2]
epsilon = 1                       # Emissivity                             [-]
albedo = 0.055                    # Albedo                                 [-]
lambda_solid = 0.5                # thermal conductivity of solid material [W/(m * K)], depending on T!
lambda_water_ice = 651            # thermal conductivity of water ice      [W/(m * T)], depending on T!
lambda_co2_ice = 0.02             # thermal conductivity of water ice      [W/(m * K)], depending on T!
a_H2O = 3.23E12                   # Sublimation Pressure Coefficient of water [Pa]
b_H2O = 6134.6                    # Sublimation Pressure Coefficient of water [K]
a_CO2 = 1.32E12                   # Sublimation Pressure Coefficient of water [Pa]
b_CO2 = 3167.8                    # Sublimation Pressure Coefficient of water [K]
#Illumination Condition/Celestial Mechanics
Period = 43400                    # Period of Rotation around Spin Axis    [s]
r_H = 1.24                           # Heliocentric Distance                  [AU]
latitude = 0                      # Latitude                               [°]
tilt = 0                          # Angle of Surface Normal to Sun         [°]
axial_tilt = 0                    # Axial Tilt/Obliquity of Spin Axis      [°]
equinox = 0                       # Time of Equinox                        [part of orbit period]
semi_major_axis = 3.4628          # Semi major axis in elliptical case     [AU]
eccentricity = 0.641              # Eccentricity of elliptical orbit       [-]
perihelion = 6.44/2               # Time of Perihelion                     [years]
orbit_period = np.sqrt(semi_major_axis**3) # Orbital Period for elliptical case     [years]

#Constants
sigma = 5.67E-8                   # Stefan-Boltzmann Constant              [W/(m^2 K^4)]
solar_constant = 1367             # Solar Constant                         [WW/m^2]
k_boltzmann = 1.38E-23            # Boltzmann's Constant                   [m^2 kg / (s^2 K)]
avogadro_constant = 6.022E23      # Avogadro constant                      [1/mol]

#MIRO MODE: Definition of MIRO-specific Parameters // Import of Tables // Calculation of Running Time
if miro_switch >= 1:
    if number_of_days <= 1000:
        r_H_tab = np.loadtxt('Import_Tables/HeliocentricDistance_1000days.txt')[(1000-number_of_days):]
    else:
        r_H_tab = np.loadtxt('Import_Tables/HeliocentricDistance_1000days.txt')
        r_H_tab = np.insert(r_H_tab, 0, np.zeros(number_of_days-1000))
        perihelion = orbit_period * (0.9058 + (number_of_days/(orbit_period*365.25*24*60*60/Period) - int(number_of_days/(orbit_period*365.25*24*60*60/Period))))
if miro_switch == 1:
    if number_of_days <= 1000:
        tilt_tab =  np.loadtxt('Import_Tables/Day_MM19032016_Verkippung_1000days.txt')[(1000-number_of_days):]
    else: 
        tilt_tab = np.loadtxt('Import_Tables/Day_MM19032016_Verkippung_1000days.txt')
        tilt_tab = np.insert(tilt_tab, 0, np.zeros(number_of_days-1000))
    t1 = int(((number_of_days - 2) * Period + 16000 - 10850) / dt )			
    k = t1 + 1	
    Temperature_Profile_DAY = np.zeros(n+1)
elif miro_switch == 2:
    if number_of_days <= 1000:
        tilt_tab =  np.loadtxt('Import_Tables/Day_MM21032016_Verkippung_1000days.txt')[(1000-number_of_days):]
    else: 
        tilt_tab =  np.loadtxt('Import_Tables/Day_MM21032016_Verkippung_1000days.txt')
        tilt_tab = np.insert(tilt_tab, 0, np.zeros(number_of_days-1000))
    Temperature_Profile_NIGHT1 = np.zeros(n+1)
    Temperature_Profile_NIGHT2 = np.zeros(n+1)	
    t2 = int(((number_of_days-1) * Period + 38600 - 10600) / dt )	
    t3 = int(((number_of_days-1) * Period + 39700 - 10600) / dt )
    k = t3 + 1


#Definition of Arrays
if dx_switch == 0:
    n = int(total_depth/min_dx)
Fourier_number = np.zeros(n+1)
Max_Fourier_number = np.zeros(k)
Energy_Increase_per_Layer = np.zeros(n+1)
E_conservation = np.zeros(k)
Total_lost_Energy = 0
Total_Input = 0
Energyloss_relative_to_totalInput = np.zeros(k)
surface_temperature = np.zeros(k)
delta_T = np.zeros(n+1)
temperature_temporary = np.zeros(n+1)
pressure = np.zeros(n+1)
pressure_co2 = np.zeros(n+1)
Lambda = []
temperature = []
VFF_pack = []
dust_ice_ratio_per_layer = []
co2_h2o_ratio_per_layer = []
j_leave = np.zeros(n+1)
j_leave_co2 = np.zeros(n+1)
j_inward = np.zeros(n+1)
j_inward_co2 = np.zeros(n+1)
outgassed_molecules_time_step = 0
outgassed_molecules_time_step_co2 = 0
outgassed_molecules_total = 0
outgassed_molecules_total_co2 = 0
outgassing_rate = []
outgassing_rate_co2 = []
VFF_agg_arr = np.zeros(n+1)
for i in range(len(VFF_agg_arr)):
    VFF_agg_arr[i] = VFF_agg_base
density_dust = np.zeros(n+1)

if dx_switch == 1:
    max_dx = min_dx + 2*(total_depth - min_dx*(n+1))/(n+1)
    dx = np.linspace(min_dx,max_dx,n+1)                     
elif dx_switch == 2:
    X = 1
    max_dx = min_dx
    while X > 0.001:
        max_dx = max_dx + min_dx
        total_depth_0 = sum(np.logspace(np.log10(min_dx),np.log10(max_dx),n+1))
        X = total_depth - total_depth_0    
    dx = np.logspace(np.log10(min_dx),np.log10(max_dx),n+1)
else:
    dx = np.ones(n+1) * min_dx
depth = []
DX = []                                          # Distance between center of layer i and center of layer underneath (i+1)
#dust_mass_in_dust_ice_layers and base_water_particle_number would need an extra version for non constant dx to work properly
dust_mass_in_dust_ice_layers = density_nucleus * min_dx * dust_ice_ratio_global / (dust_ice_ratio_global + 1)  #[kg]
dust_molecules_in_dust_ice_layers = dust_mass_in_dust_ice_layers / molar_mass_dust * avogadro_constant
water_content_per_layer = []
co2_content_per_layer = []
heat_capacity = []
eject = -1
base_water_particle_number = density_nucleus * min_dx * (1 / (dust_ice_ratio_global + 1)) * (1 - (1 / co2_h2o_ratio_global + 1)**(-1)) / molar_mass_water * avogadro_constant
base_co2_particle_number = density_nucleus * min_dx * (1 / (dust_ice_ratio_global + 1)) * (1 / co2_h2o_ratio_global + 1)**(-1) / molar_mass_co2 * avogadro_constant
for elem in range(n + 1):
    depth.append(sum(dx[:elem]))
    Lambda.append(lambda_constant)
    temperature.append(temperature_ini)
    if elem <= dust_layer_thickness-1:
        dust_ice_ratio_per_layer.append(0)
        co2_h2o_ratio_per_layer.append(0)
        water_content_per_layer.append(0)
        co2_content_per_layer.append(0)
        heat_capacity.append(heat_capacity_dust)
    else:
        dust_ice_ratio_per_layer.append(1 / (dust_ice_ratio_global + 1))
        co2_h2o_ratio_per_layer.append(((1 / co2_h2o_ratio_global) + 1)**(-1))
        water_content_per_layer.append(base_water_particle_number)
        co2_content_per_layer.append(base_co2_particle_number)
        heat_capacity.append(1 / (dust_ice_ratio_global + 1) * (1 - (1 / co2_h2o_ratio_global + 1)**(-1)) * heat_capacity_water_ice + dust_ice_ratio_global / (dust_ice_ratio_global + 1) * heat_capacity_dust + (1 / (dust_ice_ratio_global + 1)) * (1 / co2_h2o_ratio_global + 1)**(-1) * heat_capacity_co2_ice)
for elem in range(n):
    DX.append(dx[elem]/2 + dx[elem+1]/2)
day_position_store = np.zeros(k)
axial_tilt_store = np.zeros(k)
declination_store = np.zeros(k)
day_length_store = np.zeros(k)
time = np.arange(k) * dt
day_store = 0
day_start = 0
axial_tilt_factor = 1
layer_strength = np.zeros(n+1)
gravitational_pressure = 0 #Placeholder! [Pa]
for i in range(0, n):
    if depth_dependant_tensile_strength == 1:
        layer_strength[i] = depth_dependant_strength * (1 + i * dx[i] / x_0)**(-1/2) + gravitational_pressure
    else:
        layer_strength[i] = const_tensile_strength + gravitational_pressure

#The following arrays are mostly for data saving and debugging.
added_molecules = 0
added_molecules_co2 = 0
total_error = 0
total_error_co2 = 0
lost_molecules_array = []
lost_molecules_array_co2 = []
total_ejection_events = 0
ejection_times = []
complete_temperature = []
deeper_diffusion = 0
deeper_diffusion_co2 = 0
total_error_array = []
diffusion_factors = [3/6, 2/6, 1/6]
test = []
pressure_arr = []
pressure_arr_co2 = []
wcpl_complete = []
ccpl_complete = []
lambda_arr = []
heat_cap_arr = []
j_leave_complete = []
j_leave_co2_complete = []
j_inward_complete = []
j_inward_co2_complete = []
highest_pressure = np.zeros(n+1)
highest_pressure_co2 = np.zeros(n+1)
heliocentric_distance_arr = []
time_passed = 86800000 + 2 * 4340000
ejected_water = 0
VFF = []
density_test = []
p_complete = []
p_c_complete = []

if load == 1:
    with open('TPM_save_data.json') as json_file:
        data = json.load(json_file)
    temperature = data['temperature']
    water_content_per_layer = data['water content per layer']
    co2_content_per_layer = data['co2 content per layer']
    dust_ice_ratio_per_layer = data['dust ice ratio per layer']
    co2_h2o_ratio_per_layer = data['co2 h2o ratio per layer']
    heat_capacity = data['heat capacity']
    pressure = np.array(data['pressure'])
    pressure_co2 = np.array(data['pressure co2'])
    total_ejection_events = data['total ejection events']
    time_passed = data['time passed']

#debug code - delete later
total_water_molecules = sum(water_content_per_layer)
total_co2_molecules = sum(co2_content_per_layer)
print('The total number of water molecules is: ' + str(total_water_molecules))
print('The total number of CO2 molecules is: ' + str(total_co2_molecules))

#Implementation of Reduced VFF at the surface (Thilo-Function)
density_grain = density_nucleus / (VFF_pack_const * VFF_agg_base)
density = []                                                # bulk density
for elem in range(n + 1):
    if VFF_switch == 1:
        if elem < n:
            number_pebble_layer = (depth[elem+1] - dx[elem]/2) / r_agg
            if number_pebble_layer <= 1.91:
                VFF_pack.append(number_pebble_layer/1.91 * VFF_pack_const)
            else:
                VFF_pack.append(VFF_pack_const)
        else:
            VFF_pack.append(VFF_pack_const)
    else:
        VFF_pack.append(VFF_pack_const)
    density.append(VFF_pack[elem] * VFF_agg_base * density_grain)


#Definition of PEBBLE CASE
def lambda_pebble(T,x,VFF_agg):
    surface_energy_agg = VFF_agg * surface_energy_par ** (5 / 3) * (
        9 * np.pi * (1 - poisson_ratio_agg ** 2) / (r_mono * young_modulus_par)) ** (2 / 3)
    if dust_ice_ratio_per_layer[x] == 0 and co2_h2o_ratio_per_layer[x] == 0:
        lambda_agg = lambda_solid * (9 * np.pi / 4 * (
                1 - poisson_ratio_par ** 2) / young_modulus_par * surface_energy_par * r_mono ** 2) ** (
                             1 / 3) * f_1 * np.exp(f_2 * VFF_agg) / r_mono
    else:
        lambda_agg = ((1 - dust_ice_ratio_per_layer[x]) * lambda_solid + (dust_ice_ratio_per_layer[x] * (1 - co2_h2o_ratio_per_layer[x])) * (lambda_water_ice/T) + (dust_ice_ratio_per_layer[x] * co2_h2o_ratio_per_layer[x]) * lambda_co2_ice) * (9 * np.pi / 4 * (
                1 - poisson_ratio_par ** 2) / young_modulus_par * surface_energy_par * r_mono ** 2) ** (
                             1 / 3) * f_1 * np.exp(f_2 * VFF_agg) / r_mono
    lambda_net = lambda_agg * (9 * np.pi / 4 * (1 - poisson_ratio_agg ** 2) / young_modulus_agg * surface_energy_agg * r_agg ** 2) ** (1 / 3) * f_1 * np.exp(f_2 * VFF_pack[x]) / r_agg
    lambda_rad = 16 / 3 * sigma * T ** 3 * e_1 * (1 - VFF_pack[x]) / VFF_pack[x] * r_agg
    lambda_total = lambda_net + lambda_rad
    return lambda_total, lambda_net, lambda_rad

'''This function shifts the imput array A according to the number of ejected layers n 
and adds the same amount of layers at the base value of this array.'''

def ejection_array_shift(A, n, base_value):
	updated_array = []
	for i in range(n, len(A) + n):
		if i < len(A):
			updated_array.append(A[i])
		else:
			updated_array.append(base_value)
	return updated_array


'''This function updates the dust to ice ratio per layer. Always run this function before the other update function(s)'''

def update_d_i_r():
    global dust_ice_ratio_per_layer
    global co2_h2o_ratio_per_layer
    for i in range(0, n):
        if water_content_per_layer[i] == 0 and co2_content_per_layer[i] < 10**(-20):
            dust_ice_ratio_per_layer[i] = 0
        if co2_content_per_layer[i] < 10**(-20):
            co2_h2o_ratio_per_layer[i] = 0
        if water_content_per_layer[i] != 0:
            mass_ice = water_content_per_layer[i] / avogadro_constant * molar_mass_water + co2_content_per_layer[i] / avogadro_constant * molar_mass_co2
            dust_ice_ratio_per_layer[i] = mass_ice / (mass_ice + dust_mass_in_dust_ice_layers)
        if co2_h2o_ratio_per_layer[i] >= 10**(-20):
            mass_co2 = co2_content_per_layer[i] / avogadro_constant * molar_mass_co2
            co2_h2o_ratio_per_layer[i] = mass_co2 / mass_ice


'''This function updates the heat capacity. Always run 'update_d_i_r' in the main loop before the other update function(s)'''

def update_heat_capacity():
    global heat_capacity
    for i in range(0, n+1):
        if dust_ice_ratio_per_layer[i] == 0 and co2_h2o_ratio_per_layer[i] == 0:
            heat_capacity[i] = heat_capacity_dust
        elif dust_ice_ratio_per_layer != 0 and co2_h2o_ratio_per_layer[i] == 0:
            heat_capacity[i] = heat_capacity_dust * (1 - dust_ice_ratio_per_layer[i]) + heat_capacity_water_ice * dust_ice_ratio_per_layer[i]
        elif dust_ice_ratio_per_layer == 0 and co2_h2o_ratio_per_layer[i] != 0:
            print('This case should not be possible')
        else:
            heat_capacity[i] = heat_capacity_dust * (1 - dust_ice_ratio_per_layer[i]) + heat_capacity_water_ice * (dust_ice_ratio_per_layer[i] * (1 - co2_h2o_ratio_per_layer[i])) + heat_capacity_co2_ice * (dust_ice_ratio_per_layer[i] * co2_h2o_ratio_per_layer[i])

'''This function updates the volume filling factor. Always run 'update_d_i_r' in the main loop before the other update function(s)'''

'''def update_VFF_agg():
    global VFF_agg_arr
    for i in range(0, n+1):
        VFF_agg_arr[i] = VFF_agg_base * (dust_molecules_in_dust_ice_layers + water_content_per_layer[i] + co2_content_per_layer[i])/(dust_molecules_in_dust_ice_layers + base_water_particle_number + base_co2_particle_number)
        density_dust[i] = density_nucleus / (VFF_agg_arr[i] * VFF_pack[i]) * 1 / (1 - dust_ice_ratio_per_layer[i]) - density_water_ice * ((dust_ice_ratio_per_layer[i] * (1 - co2_h2o_ratio_per_layer[i])**(-1))/ (1 - dust_ice_ratio_per_layer[i])) - density_co2_ice * ((dust_ice_ratio_per_layer[i] * co2_h2o_ratio_per_layer[i])/ (1 - dust_ice_ratio_per_layer[i]))

update_VFF_agg()'''
      
######################################
########Start Core Programme##########
######################################

#Time Loop
for j in tqdm(range(0, k)): 
    #Calculation of elliptical orbit 
    t = (j * dt + time_passed) / ( 60 * 60 * 24 * 365.25)       # [years]
    if elliptical_orbit == 1 or miro_switch >= 1 and r_H_tab[day_counter]==0 :      
        M = 2 * np.pi / orbit_period * (t - perihelion)
        x = (eccentricity * np.sin(M)) / (1 - eccentricity * np.cos(M))
        E = M + x * (1 - 1/2 * x ** 2)
        E_old = E
        X = 1
        
        while abs(X) > 0.0001:
            E_old = E
            E = E - ((E- eccentricity * np.sin(E) - M) / (1 - eccentricity * np.cos(E)))
            X = (E - E_old) / E_old
        v = 2 * np.arctan(np.sqrt((1 + eccentricity) / (1 - eccentricity)) * np.tan(1/2 * E))
    
        r_H = (semi_major_axis * (1 - eccentricity ** 2)) / (1 + eccentricity * np.cos(v))
        
        if miro_switch >= 1 and r_H_tab[day_counter]==0 :
            r_H = r_H_tab[day_counter]
    
    heliocentric_distance_arr.append(r_H)
    #Define Day-Night-Cycle
    if day_night == 1 and miro_switch == 0:
        declination = np.sin(2 * np.pi * (1 + t/orbit_period - equinox)) * axial_tilt * np.pi / 180
        axial_tilt_factor = np.cos(latitude * 2 * np.pi / 360 - declination)
        axial_tilt_store[j] = axial_tilt_factor
        if np.tan(declination) * np.tan(latitude * np.pi / 180) < -1:
            day_length = 0
        elif np.tan(declination) * np.tan(latitude * np.pi / 180) > 1:
            day_length = 2* Period
        else:
            day_length = (2 * np.arcsin(np.tan(declination) * np.tan(latitude * np.pi / 180)) + np.pi) * Period / (2* np.pi)
        if day_store < 1 and Period/2 <= day_length <= Period:
            day_position = math.sin((j * dt - day_start) * 2. * math.pi / (day_length * 2))
            day_store = j * dt / Period - day_counter
        elif day_store < 1 and 0 < day_length < Period/2:
            if day_store <= day_length/Period:
                day_position = math.sin((j * dt - day_start) * 2. * math.pi / (day_length * 2))
            else:
                day_position = 0
            day_store = j * dt / Period - day_counter
        elif day_store < 1 and day_length > Period:
            day_position = 1
            day_store = j * dt / Period - day_counter
        elif day_store < 1 and day_length == 0:
            day_position = 0
            day_store = j * dt / Period - day_counter
        else:
            day_counter += 1
            day_store = 0
            day_start = j * dt
            if 0 < day_length < Period:
                day_position = math.sin((j * dt - day_start) * 2. * math.pi / (day_length * 2))
            elif day_length >= Period:
                day_position = 1
            else:
                day_position = 0
            if Period/dt < j:
                print("Day:", day_counter, "r_H:", r_H, "tilt:", round(latitude * 2 * np.pi / 360 - declination), "F_max:", round(max(Max_Fourier_number[j-int(Period/dt):j]),2))
            else:
                print("Day:", day_counter, "r_H:", r_H, "tilt:", round(latitude * 2 * np.pi / 360 - declination), "F_max:", round(max(Max_Fourier_number),2))
        declination_store[j] = declination
        day_length_store[j] = day_length   
        #print(declination, day_length,day_position,day_store)
    elif day_night == 1 and miro_switch >= 1:
        day_length = Period / 2
        if day_store < 1:
            day_position = math.sin((j * dt) * 2. * math.pi / (day_length * 2) + np.pi/2)
            day_store = j * dt / Period - day_counter
        else:
            day_counter += 1
            day_store = 0
            day_position = math.sin((j * dt) * 2. * math.pi / (day_length * 2) + np.pi/2)
            if Period/dt < j:
                print("Day:", day_counter, "r_H:", r_H, "tilt:", round(tilt,2), "F_max:", round(max(Max_Fourier_number[j-int(Period/dt):j]),2))
            else:
                print("Day:", day_counter, "r_H:", r_H, "tilt:", round(tilt,2), "F_max:", round(max(Max_Fourier_number),2))
    else:
        day_position = 1
        
    if day_position < 0:
        day_position = 0           
         
    day_position_store[j] = day_position
    

    if miro_switch >= 1:
        r_H = r_H_tab[day_counter] 
        tilt = tilt_tab[day_counter]
    

    #Define Energie Source/Sink
    if miro_switch >= 1:
        Solar_Intensity = solar_constant / r_H ** 2 * np.cos(tilt * np.pi / 180) * (1 - albedo)
    else:
        Solar_Intensity = solar_constant / r_H ** 2 * (1 - albedo)
    if pebble_case == 1:
        Lambda[0] = lambda_pebble( temperature[1] + (temperature[0]-temperature[1])/DX[0] * 1/2 * dx[1] ,0 ,VFF_agg_arr[0])[0]
        #lambda_arr.append(lambda_pebble( temperature[1] + (temperature[0]-temperature[1])/DX[0] * 1/2 * dx[1] ,0 ,VFF_agg_arr[0])[2])
        #Lambda[n] = lambda_pebble( temperature[n+1] + (temperature[n]-temperature[n+1])/DX[n] * 1/2 * dx[n+1], n, VFF_agg_arr[n])[0]
    if axial_tilt_factor > 0:
        E_In = Solar_Intensity * dt * day_position * axial_tilt_factor            # [J/(m^2)]
    else:
        E_In = 0
    E_Rad = - sigma * epsilon * temperature[0]**4 * dt                            # [J/(m^2)]	
    E_Cond = Lambda[0] * (temperature[1] - temperature[0]) / DX[0] * dt
    E_Energy_Increase = E_In + E_Rad + E_Cond
    delta_T[0] = E_Energy_Increase / (specific_heat_capacity * density[0] * dx[0])
    Energy_Increase_per_Layer[0] = specific_heat_capacity * density[0] * dx[0] * delta_T[0]     # [J/m^2]
   
    #Interior Loop
    for i in range(0,n+1):
        if pebble_case == 1:
            if temperature[i] >= 75:
                j_leave[i] = a_H2O * np.exp(- b_H2O / temperature[i]) * np.sqrt(m_H2O / (2 * np.pi * k_boltzmann * temperature[i])) * (1 + (i * dx[i]) / b) ** (-1)  # [kg/(m^2 * s)]
                if j_leave[i] * dt > (water_content_per_layer[i] / avogadro_constant) * molar_mass_water:
                    j_leave[i] = water_content_per_layer[i] / (avogadro_constant * dt) * molar_mass_water
            else:
                j_leave[i] = 0
            if temperature[i] >= 52 and co2_content_per_layer[i] > 10**(-20):
                j_leave_co2[i] = a_CO2 * np.exp(- b_CO2 / temperature[i]) * np.sqrt(
                    m_CO2 / (2 * np.pi * k_boltzmann * temperature[i])) * (1 + (i * dx[i]) / b) ** (
                                     -1)  # [kg/(m^2 * s)]
                if j_leave_co2[i] * dt > (co2_content_per_layer[i] / avogadro_constant) * molar_mass_co2:
                    j_leave_co2[i] = co2_content_per_layer[i] / (avogadro_constant * dt) * molar_mass_co2
            else:
                j_leave_co2[i] = 0

    j_inward = np.zeros(n+1)
    j_inward_co2 = np.zeros(n+1)

    for i in range (0, n+1):
        '''if i > 0:
            temperature_temporary[i] = temperature[i] + delta_T[i]'''
        for m in range(0, len(diffusion_factors)):
            if i + len(diffusion_factors) < n:
                j_inward[i+1+m] += j_leave[i]/2 * diffusion_factors[m]
                j_inward_co2[i+1+m] += j_leave_co2[i]/2 * diffusion_factors[m]
            else:
                if i + m < n:
                    j_inward[i+1+m] += j_leave[i]/2 * diffusion_factors[m]
                    j_inward_co2[i + 1 + m] += j_leave_co2[i]/ 2 * diffusion_factors[m]
                else:
                    deeper_diffusion += j_leave[i]/2 * diffusion_factors[m]
                    deeper_diffusion_co2 += j_leave_co2[i]/2 * diffusion_factors[m]

    '''j_leave_complete.append(j_leave[:].tolist())
    j_leave_co2_complete.append(j_leave_co2[:].tolist())
    j_inward_complete.append(j_inward[:].tolist())
    j_inward_co2_complete.append(j_inward_co2[:].tolist())'''

    for i in range(1, n):
        if pebble_case == 1:
            Lambda[i] = lambda_pebble( temperature[i+1] + (temperature[i]-temperature[i+1])/DX[i] * 1/2 * dx[i+1] ,i ,VFF_agg_arr[i])[0]
        #Standard Thermal Diffusivity Equation
        delta_T[i] = ((((temperature[i-1] - temperature[i]) * Lambda[i-1] / (DX[i-1])) \
        - ((temperature[i] - temperature[i + 1]) * Lambda[i] / (DX[i]) )) / dx[i]) * dt / (density[i] * heat_capacity[i]) - 3 * VFF_pack[i] / r_agg * (j_leave[i] - j_inward[i]) * latent_heat_water * dt / (density[i] * heat_capacity[i]) -  3 * VFF_pack[i] / r_agg * (j_leave_co2[i] - j_inward_co2[i]) * latent_heat_co2 * dt / (density[i] * heat_capacity[i]) # [K]
        Fourier_number[i] = Lambda[i] / (density[i] * specific_heat_capacity) * dt / dx[i]**2       # [-]
        Energy_Increase_per_Layer[i] = specific_heat_capacity * density[i] * dx[i] * delta_T[i]     # [J/m^2] 
    surface_temperature[j] = temperature[0]

    total_error += sum(j_leave)/2 - sum(j_inward) - deeper_diffusion
    total_error_co2 += sum(j_leave_co2) / 2 - sum(j_inward_co2) - deeper_diffusion_co2

	#Secure Numerical Stability: End Simulation if Fourier Number > 0.5
    Max_Fourier_number[j] = max(Fourier_number)
    if Fourier_break == 1 and max(Fourier_number) > 0.5:
        print('Fourier number > 0.5 !')
        break

    #Set Energy Loss per Timestep = 0 -> Differential Counting of Energy Loss
    Energy_Increase_Total_per_time_Step = 0                                                   # [J/m^2]
       
	#Update Temperature and Energy Loss Counting Array
    for i in range(0, n+1):
        temperature[i] = temperature[i] + delta_T[i]                                          # [K]
        Energy_Increase_Total_per_time_Step = Energy_Increase_Total_per_time_Step + Energy_Increase_per_Layer[i]    # [J/m^2]
        water_content_per_layer[i] += (j_inward[i] - j_leave[i]) * dt / molar_mass_water * avogadro_constant
        co2_content_per_layer[i] += (j_inward_co2[i] - j_leave_co2[i]) * dt / molar_mass_co2 * avogadro_constant
        outgassed_molecules_time_step += j_leave[i]/2 * dt / molar_mass_water * avogadro_constant
        outgassed_molecules_time_step_co2 += j_leave_co2[i]/2 * dt / molar_mass_co2 * avogadro_constant
        if j_leave[i] == 0:
            pressure[i] = 0
        else:
            pressure[i] = a_H2O * np.exp(- b_H2O / temperature[i]) * (1 - (1 + i * dx[i] / b)**(-1))
        if j_leave_co2[i] == 0:
            pressure_co2[i] = 0
        else:
            pressure_co2[i] = a_CO2 * np.exp(- b_CO2 / temperature[i]) * (1 - (1 + i * dx[i] / b)**(-1))
        '''if water_content_per_layer[i] == 0:
            pressure[i] = 0
        if co2_content_per_layer[i] == 0:
            pressure_co2[i] = 0'''
        if water_content_per_layer[i] == 0 and pressure[i] != 0:
            pressure_arr.append([highest_pressure[i], (i + total_ejection_events), j*dt, temperature[i]])
        if co2_content_per_layer[i] == 0 and pressure_co2[i] != 0:
            pressure_arr_co2.append([highest_pressure_co2[i], (i + total_ejection_events), j*dt, temperature[i]])

    if ejection == 1:
        for i in range(0, n):
            #check if that should also be n+1
            if pressure[i] + pressure_co2[i] > layer_strength[i]:
                eject = i

    for i in range(0,n+1):
        if pressure[i] > highest_pressure[i]:
            highest_pressure[i] = pressure[i]
        if pressure_co2[i] > highest_pressure_co2[i]:
            highest_pressure_co2[i] = pressure_co2[i]

    if eject != -1:
        print(eject) #debbug print statement
        for i in range(0, eject):
            ejected_water += water_content_per_layer[i]
        temperature = ejection_array_shift(temperature, eject, temperature_ini)
        ejection_times.append([j * dt, eject, pressure[eject], pressure_co2[eject]])
        pressure = ejection_array_shift(pressure, eject, 0)
        pressure_co2 = ejection_array_shift(pressure_co2, eject, 0)
        water_content_per_layer = ejection_array_shift(water_content_per_layer, eject, base_water_particle_number)
        co2_content_per_layer = ejection_array_shift(co2_content_per_layer, eject, base_co2_particle_number)
        dust_ice_ratio_per_layer = ejection_array_shift(dust_ice_ratio_per_layer, eject, (1 / (dust_ice_ratio_global + 1)))
        co2_h2o_ratio_per_layer = ejection_array_shift(co2_h2o_ratio_per_layer, eject, (((1 / co2_h2o_ratio_global) + 1)**(-1)))
        total_ejection_events += eject
        highest_pressure = np.zeros(n+1)
        highest_pressure_co2 = np.zeros(n+1)

    #MIRO MODE: Store of Temperature Profiles
    if miro_switch == 1 and j == t1:
        Temperature_Profile_DAY = temperature
    elif miro_switch == 2 and j == t2:
        Temperature_Profile_NIGHT1 = temperature
    elif miro_switch == 2 and j == t3:
        Temperature_Profile_NIGHT2 = temperature

    outgassed_molecules_total += outgassed_molecules_time_step
    outgassed_molecules_total_co2 += outgassed_molecules_time_step_co2
    outgassing_rate.append(outgassed_molecules_time_step)
    outgassing_rate_co2.append((outgassed_molecules_time_step_co2))
    update_d_i_r()
    update_heat_capacity()
    #update_VFF_agg()
    outgassed_molecules_time_step = 0
    outgassed_molecules_time_step_co2 = 0

    if eject != 0:
        for i in range(n - eject + 1, n + 1):
            added_molecules += water_content_per_layer[i]
            added_molecules_co2 += co2_content_per_layer[i]
    eject = -1
        
    #Conservation of Energy Check
    E_conservation[j] = Energy_Increase_Total_per_time_Step - E_In - E_Rad                    # [J/m^2]
    Total_lost_Energy = Total_lost_Energy + E_conservation[j]  
    Total_Input = Total_Input + E_In                                                          # [J/m^2]  
    if Total_Input > 0:
        Energyloss_relative_to_totalInput[j] = Total_lost_Energy/ Total_Input
    #print(Total_lost_Energy/ Total_Input)

    #print(Total_lost_Energy/ Total_Input)

    lost_molecules_array.append(sum(water_content_per_layer, outgassed_molecules_total) - total_water_molecules - added_molecules + ejected_water)
    lost_molecules_array_co2.append(sum(co2_content_per_layer, outgassed_molecules_total_co2) - total_co2_molecules - added_molecules_co2)
    if j % 434 == 0:
        #turn this on for longer runs since the RAM could overflow
        wcpl_complete.append(dust_ice_ratio_per_layer[:])
        ccpl_complete.append(co2_h2o_ratio_per_layer[:])
        complete_temperature.append(temperature[:])
    '''wcpl_complete.append(dust_ice_ratio_per_layer[:])
    ccpl_complete.append(co2_h2o_ratio_per_layer[:])
    complete_temperature.append(temperature[:])'''
    #lambda_arr.append(Lambda[:])
    #heat_cap_arr.append(heat_capacity[:])
    #VFF.append(VFF_agg_arr[:].tolist())
    #density_test.append(density_dust[:].tolist())
    '''p_complete.append(pressure[:].tolist())
    p_c_complete.append(pressure_co2[:].tolist())
    total_error_array.append(total_error)'''

    #Debug Code - delete later
    if j % 4340   == 0:
        print('The number of water molecules lost after ' + str(j * dt) + ' seconds is: ' + str(sum(water_content_per_layer, outgassed_molecules_total) - total_water_molecules - added_molecules + ejected_water))
        print('The number of co2 molecules lost after ' + str(j * dt) + ' seconds is: ' + str(
            sum(co2_content_per_layer, outgassed_molecules_total_co2) - total_co2_molecules - added_molecules_co2))
        print(temperature)
        #print(water_content_per_layer)
        #print(co2_content_per_layer)
        print(dust_ice_ratio_per_layer)
        print(co2_h2o_ratio_per_layer)
        #print(heat_capacity)
        #print(Lambda)
        #print(VFF_agg_arr)
        #print(delta_T[0])
        print(total_error)
        print(r_H)

if Total_Input > 0:
    print('Energy Loss rel. to Total Input: ', Total_lost_Energy/ Total_Input)
else:
    print('No Input Energy! Absolute Energy Loss [J]: ', Total_lost_Energy)
print('Fourier Number (Min, Max): ', min(Max_Fourier_number), max(Max_Fourier_number))       
#print('Temperature Array:') 
#print(temperature)

if save == 1:
    if total_ejection_events > 0:
        data = {'temperature': temperature, 'water content per layer': water_content_per_layer, 'co2 content per layer' :co2_content_per_layer, 'dust ice ratio per layer': dust_ice_ratio_per_layer, 'co2 h2o ratio per layer': co2_h2o_ratio_per_layer, 'heat capacity': heat_capacity, 'pressure': pressure, 'pressure co2': pressure_co2, 'total ejection events': total_ejection_events, 'time passed': (k * dt + time_passed)}
    else:
        data = {'temperature': temperature, 'water content per layer': water_content_per_layer, 'co2 content per layer' :co2_content_per_layer,
                'dust ice ratio per layer': dust_ice_ratio_per_layer, 'co2 h2o ratio per layer': co2_h2o_ratio_per_layer, 'heat capacity': heat_capacity,
                'pressure': pressure.tolist(), 'pressure co2': pressure_co2.tolist(), 'total ejection events': total_ejection_events,
                'time passed': (k * dt + time_passed)}
    with open('TPM_save_data.json', 'w') as outfile:
        json.dump(data, outfile)
        
#Data Save code
'''for i in range(len(density_test)):
    density_test[i] = density_test[i].tolist()'''
dict = {'Temperature': complete_temperature, 'Outgassing Rate': outgassing_rate,'Outgassing Rate CO2': outgassing_rate_co2, 'Lost Molecules': lost_molecules_array, 'Ejection Events': ejection_times, 'Total Molecules': total_water_molecules, 'Total Ejection Events': total_ejection_events, 'Water Content Per Layer': wcpl_complete, 'CO2 Content Per Layer': ccpl_complete,'Pressure': highest_pressure.tolist(), 'Pressure CO2': highest_pressure_co2.tolist(), 'Heliocentric Distance': heliocentric_distance_arr, 'P2': pressure_arr, 'PCO2': pressure_arr_co2}
#dict = {'Temperature': complete_temperature, 'Outgassing Rate': outgassing_rate,'Outgassing Rate CO2': outgassing_rate_co2, 'Lost Molecules': lost_molecules_array, 'Ejection Events': ejection_times, 'Total Molecules': total_water_molecules, 'Total Ejection Events': total_ejection_events, 'Water Content Per Layer': wcpl_complete, 'CO2 Content Per Layer': ccpl_complete,'Pressure': p_complete, 'Pressure CO2': p_c_complete, 'Heliocentric Distance': heliocentric_distance_arr, 'P2': pressure_arr, 'PCO2': pressure_arr_co2, 'JL': j_leave_complete, 'JI': j_inward_complete, 'JLC': j_leave_co2_complete, 'JIC': j_inward_co2_complete, 'HP': highest_pressure.tolist(), 'HPCO2': highest_pressure_co2.tolist()}
ELRTI = []
for i in range (0, len(Energyloss_relative_to_totalInput)):
    ELRTI.append(Energyloss_relative_to_totalInput[i])
#dict_2 = {'Temperature': complete_temperature,'J Leave': j_leave_complete, 'J Inward': j_inward_complete, 'Water Content Per Layer': wcpl_complete, 'CO2 Content Per Layer': ccpl_complete,'Pressure': highest_pressure.tolist(), 'Pressure CO2': highest_pressure_co2.tolist(), 'P2': pressure_arr, 'PCO2': pressure_arr_co2}

with open('BT_run_new_coeff_start_2200_2400_ts_0_05.json', 'w') as outfile:
    json.dump(dict, outfile)

print('done')
# Plot part
# Make Plots pretty: %config InlineBackend.figure_format = 'retina' 
x_1 = time
y_1 = day_position_store * max(abs(Energyloss_relative_to_totalInput))
plot.plot(x_1,y_1)
x = time
y = Energyloss_relative_to_totalInput
plot.plot(x, y) 
# naming the x,y axis 
plot.xlabel('Time [s]') 
plot.ylabel('Energy loss relative to total input') 
#plot.xticks(np.arange(0,j*dt,step=Period/2))
#plot.grid(which='major',axis='x')
# giving a title to my graph 
plot.title('Energy Loss Analysis!') 
# function to show the plot 
plot.show()    
  

if miro_switch < 1:
    x_1 = depth
    y_1 = temperature
    plot.plot(x_1,y_1) 
    # naming the x,y axis 
    plot.xlabel('Depth [m]') 
    plot.ylabel('Temperature [K]') 
    # function to show the plot 
    plot.show()   
    
    x_2 = time
    y_2 = surface_temperature
    plot.plot(x_2,y_2)
    plot.xlabel('Time [s]') 
    plot.ylabel('Surface Temperature [K]') 
        
# MIRO MDOE: Output
timestemp = date.today().strftime('%Y_%m_%d') + "_" + DateTime.now().strftime('%H_%M_%S') 
if miro_switch >= 1:
    plot.figure(figsize=(12, 6), dpi=180)
    if miro_switch == 1:
        plot.title("Temperature Profile - DAY - after " + str(number_of_days) + " days", fontsize=16)
        plot.plot(depth+np.ones(n+1)*min_dx, Temperature_Profile_DAY)
        plot.xlabel('Depth [m]', fontsize=16)
        plot.ylabel('Temperature [K]', fontsize=16)
        plot.xscale("log")
        if Output_switch == 1:
            if pebble_case == 1:
                name = str(number_of_days) + "days_pebble_n_" + str(n) + "_" + str(timestemp)
                plot.savefig("Plots/TPM_Temperature_Depth_MIRO_DAY_" + name + ".png")
                with open("Output_Tables/TPM_Temperature_Depth_MIRO_DAY_" + name + ".txt", "w") as file:
                    for item in Temperature_Profile_DAY:
                        file.write("%s\n" % item)
            else:
                name = str(number_of_days) + "days_const_n_" + str(n) + "_" + str(timestemp)
                plot.savefig("Plots/TPM_Temperature_Depth_MIRO_DAY_" + name + ".png")
                with open("Output_Tables/TPM_Temperature_Depth_MIRO_DAY_" + name + ".txt", "w") as file:
                    for item in Temperature_Profile_DAY:
                        file.write("%s\n" % item)
            
    else:
        plot.plot(depth+np.ones(n+1)*min_dx, Temperature_Profile_NIGHT1, "-r", label='NIGHT1')
        plot.xlabel('Depth [m]', fontsize=16)
        plot.ylabel('Temperature [K]', fontsize=16)
        plot.title("Temperature Profile - NIGHT - after " + str(number_of_days) + " days", fontsize=16)
        plot.xscale("log")
        plot.plot(depth+np.ones(n+1)*min_dx, Temperature_Profile_NIGHT2, "--b", label='NIGHT2')
        plot.legend()
        if Output_switch == 1:
            if pebble_case == 1:
                name = str(number_of_days) + "days_pebble_n_" + str(n) + "_" + str(timestemp)
                plot.savefig("Plots/TPM_Temperature_Depth_MIRO_NIGHT_" + name + ".png")
                with open("Output_Tables/TPM_Temperature_Depth_MIRO_NIGHT1_" + name + ".txt", "w") as file:
                    for item in Temperature_Profile_NIGHT1:
                        file.write("%s\n" % item)
                with open("Output_Tables/TPM_Temperature_Depth_MIRO_NIGHT2_" + name + ".txt", "w") as file:
                    for item in Temperature_Profile_NIGHT2:
                        file.write("%s\n" % item)
            else:
                name = str(number_of_days) + "days_const_n_" + str(n) + "_" + str(timestemp)
                plot.savefig("Plots/TPM_Temperature_Depth_MIRO_NIGHT_" + name + ".png")
                with open("Output_Tables/TPM_Temperature_Depth_MIRO_NIGHT1_" + name + ".txt", "w") as file:
                    for item in Temperature_Profile_NIGHT1:
                        file.write("%s\n" % item)
                with open("Output_Tables/TPM_Temperature_Depth_MIRO_NIGHT2_" + name + ".txt", "w") as file:
                    for item in Temperature_Profile_NIGHT2:
                        file.write("%s\n" % item)
    if dx_switch > 0 and Output_switch == 1:
        name = str(number_of_days) + "days_n_" + str(n) + "_" + str(timestemp)
        with open("Output_Tables/TPM_dx_Depth_MIRO_" + name + ".txt", "w") as file:
                for item in depth:
                    file.write("%s\n" % item)

    plot.show()
    
if Output_switch == 1:
    with open("Output_Tables/Simulation_information_" + timestemp + ".txt", "w") as file:
        file.write("Simulation Information")
        file.write("\nSettings")
        file.write("\nDay Night Cycle (1=ON,0= constant illumination): " + str(day_night))
        file.write("\nMIRO switch (0 = OFF, 1 = DAY, 2 = NIGHT (both night cases together)): " + str(miro_switch))
        file.write("\nPebble Case (1=ON,0=OFF=no temperature dependency): " + str(pebble_case))
        file.write("\nElliptical Orbit (1=ON,0=OFF): " + str(elliptical_orbit))
        file.write("\nVariable dx (linear == 1, log == 2, else constant min_dx): " + str(dx_switch))
        file.write("\nReduced VFF at surface (1=ON,0=OFF): " + str(VFF_switch))
        file.write("\nFourier_break (If == 1: Break of run if Fourier number > 0.5, else: only printed): " + str(Fourier_break))
        file.write("\nMaximum Fourier Number (-):" + str(max(Max_Fourier_number)))
        if Total_Input > 0:
            file.write("\nEnergy Loss rel. to Total Input (-):" + str(Total_lost_Energy/ Total_Input))
        else:
            file.write("\nNo Input Energy! Absolute Energy Loss (J):" + str(Total_lost_Energy))
        file.write("\nSimulation Parameters")
        file.write("\n(Minimal) Depth step (m):" + str(min_dx))
        file.write("\nTime step (s):" + str(dt))
        file.write("\nNumber of layers(-):" + str(n))
        file.write("\nNumber of time steps(-):" + str(k))
        file.write("\nTotal depth (m):" + str(total_depth))
        file.write("\nNumber of Comet Days (-):" + str(number_of_days))
        file.write("\nMaterial Properties")
        file.write("\nPebble size (m):" + str(r_agg))
        file.write("\nParticle size (m):" + str(r_mono))
        file.write("\nLambda Constant (W/(K*m)):" + str(lambda_constant))
        file.write("\nNucleus Density (kg/m^3):" + str(density_nucleus))
        file.write("\ne_1 (-):" + str(e_1))
        file.write("\nVFF_pack (-):" + str(VFF_pack_const))
        file.write("\nVFF_agg (-):" + str(VFF_agg))
        file.write("\nPoissons Ratio particle (-):" + str(poisson_ratio_par))
        file.write("\nPoissons Ratio agg (-):" + str(poisson_ratio_agg))
        file.write("\nYoungs Modulus particle (Pa):" + str(young_modulus_par))
        file.write("\nYoungs Modulus agg (Pa):" + str(young_modulus_agg))
        file.write("\nf_1 (-):" + str(f_1))
        file.write("\nf_2 (-):" + str(f_2))
        file.write("\nSurface Energy particle (J/m^2):" + str(surface_energy_par))
        file.write("\nThermal Properties")
        file.write("\nInitial Temperature (K):" + str(temperature_ini))
        file.write("\nEpsilon (-):" + str(epsilon))
        file.write("\nAlbedo (-):" + str(albedo))
        file.write("\nLambda of Material/Particle (W/(K*m)):" + str(lambda_solid))
        file.write("\nheat capacity: " + str(specific_heat_capacity))
        file.write("\nIllumination Condition/Celestial Mechanics")
        file.write("\nHeliocentric distance (AU):" + str(r_H))
        file.write("\nLatitude (0° = equator, 90° = pole):" + str(latitude))
        file.write("\nRotation period (s):" + str(Period))
        file.write("\nTilt of Surface (if no usage of latitude) (°):" + str(tilt))
        file.write("\nSemi Major Axis (AU):" + str(semi_major_axis))
        file.write("\nEccentricity (-):" + str(eccentricity))
        file.write("\nTime of Perihelion (Years):" + str(perihelion))
        file.write("\nOrbital Period (Years):" + str(orbit_period))

