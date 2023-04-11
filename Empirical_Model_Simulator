import random
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from abc import ABC, abstractmethod
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

################################ Part 1: Input   ############################

########### Example usage for Litho Track Simulator

# ###Settings
# parameter_ranges = [(2000, 8000), (30, 60), (90, 130), (60, 180), (50, 150), (0.5, 2.5), (10, 100), (30, 90), (5, 50), (20, 40), (0.26, 1)]
# parameter_names = ['Spin_Speed', 'Spin_Time', 'Hot_Plate_Temperature', 'Hot_Plate_Time', 'Developer_Dispense_Volume', 'Photoresist_Thickness', 'Exposure_Dose', 'Develop_Time', 'Photoresist_Viscosity', 'Photoresist_Solids_Content', 'Developer_Concentration']
# performance_names = ['Critical_Dimension_Uniformity', 'Line_Edge_Roughness', 'Sensitivity', 'Resolution']
# num_doe = 50
# output_csv = 'Litho_Process.csv' #normalized data
# output_raw_csv = 'Litho_Process_raw.csv' #raw data
# output_parameter_png = 'Litho_parameters.png'
# output_performance_png = 'Litho_performance.png'

# ###Function
# def Process_model(Spin_Speed, Spin_Time, Hot_Plate_Temperature, Hot_Plate_Time, Developer_Dispense_Volume, Photoresist_Thickness, Exposure_Dose, Develop_Time, Photoresist_Viscosity, Photoresist_Solids_Content, Developer_Concentration):
    # Critical_Dimension_Uniformity = 100 - (0.01 * Spin_Speed) - (0.1 * Hot_Plate_Temperature) + (0.05 * Developer_Dispense_Volume) + (0.1 * Exposure_Dose)
    # Line_Edge_Roughness = 10 - (0.001 * Spin_Speed) + (0.1 * Photoresist_Thickness) + (0.01 * Develop_Time)
    # Sensitivity = 50 - (0.1 * Photoresist_Viscosity) + (0.05 * Photoresist_Solids_Content) - (0.01 * Developer_Concentration)
    # Resolution = 100 - (0.1 * Exposure_Dose) + (0.05 * Develop_Time) - (0.01 * Photoresist_Thickness)
    # return [Critical_Dimension_Uniformity, Line_Edge_Roughness, Sensitivity, Resolution]

############ Example usage for CMP Simulator

# ###Settings
# parameter_ranges = [(30, 150), (30, 150), (100, 400), (100, 400), (50, 200), (60, 600), (2, 12), (30, 150), (1, 10), (0.5, 5), (40, 80)]
# parameter_names = ['Platen_Speed', 'Carrier_Speed', 'Downforce', 'Slurry_Flow_Rate', 'Retaining_Ring_Pressure', 'Polishing_Time', 'Slurry_pH', 'Pad_Conditioning', 'Abrasive_Concentration', 'Oxidizer_Concentration', 'Pad_Hardness']
# performance_names = ['Removal_Rate', 'Planarity', 'Surface_Roughness', 'Defect_Density']
# num_doe = 50
# output_csv = 'CMP_Process.csv' #normalized data
# output_raw_csv = 'CMP_Process_raw.csv' #raw data
# output_parameter_png = 'CMP_parameters.png'
# output_performance_png = 'CMP_performance.png'

# ###Function
# def Process_model(Platen_Speed, Carrier_Speed, Downforce, Slurry_Flow_Rate, Retaining_Ring_Pressure, Polishing_Time, Slurry_pH, Pad_Conditioning, Abrasive_Concentration, Oxidizer_Concentration, Pad_Hardness):
    # Removal_Rate = (Downforce * Slurry_Flow_Rate * Abrasive_Concentration) / (100 * Platen_Speed * Carrier_Speed)
    # Planarity = (Retaining_Ring_Pressure * Platen_Speed) / (100 * Downforce)
    # Surface_Roughness = 100 - (0.1 * Platen_Speed) + (0.05 * Pad_Conditioning) - (0.01 * Pad_Hardness)
    # Defect_Density = 1000 * (1 - (0.001 * Downforce * Slurry_Flow_Rate * Abrasive_Concentration / 1000)) + (0.1 * Oxidizer_Concentration)
    # return [Removal_Rate, Planarity, Surface_Roughness, Defect_Density]


#############  Example usage for Etch Simulator

# ###Settings
# parameter_ranges = [(100, 2000), (1, 200), (10, 500), (5, 50), (-100, -1000), (10, 300), (0.1, 10), (0, 100), (5, 100), (100, 1000), (1, 10)]
# parameter_names = ['RF_Power', 'Chamber_Pressure', 'Gas_Flow_Rate', 'Electrode_Spacing', 'Bias_Voltage', 'Etch_Time', 'Gas_Mixture_Ratio', 'Substrate_Temperature', 'Barrier_Layer_Thickness', 'Mask_Thickness', 'Etch_Selectivity']
# performance_names = ['Etch_Rate', 'Etch_Uniformity', 'Feature_Profile', 'Critical_Dimension']
# num_doe = 50
# output_csv = 'Etch_Process.csv' #normalized data
# output_raw_csv = 'Etch_Process_raw.csv' #raw data
# output_parameter_png = 'Etch_parameters.png'
# output_performance_png = 'Etch_performance.png'

# ###Function
# def Process_model(RF_Power, Chamber_Pressure, Gas_Flow_Rate, Electrode_Spacing, Bias_Voltage, Etch_Time, Gas_Mixture_Ratio, Substrate_Temperature, Barrier_Layer_Thickness, Mask_Thickness, Etch_Selectivity):
    # Etch_Rate = (RF_Power * Gas_Flow_Rate * Etch_Time * Gas_Mixture_Ratio * Bias_Voltage) / (1000000 * Chamber_Pressure * Electrode_Spacing)
    # Etch_Uniformity = 100 - (0.01 * RF_Power) - (0.1 * Chamber_Pressure) + (0.05 * Gas_Flow_Rate) + (0.1 * Electrode_Spacing) + (0.01 * Bias_Voltage)
    # Feature_Profile = 90 - (0.01 * Barrier_Layer_Thickness) - (0.1 * Mask_Thickness) + (0.05 * Etch_Selectivity)
    # Critical_Dimension = 100 - (0.1 * Etch_Time) + (0.05 * Substrate_Temperature) - (0.01 * Gas_Mixture_Ratio)
    # return [Etch_Rate, Etch_Uniformity, Feature_Profile, Critical_Dimension]


#############  Example usage for Device1 Simulator

# ###Settings
# parameter_ranges = [(5, 50), (10, 100), (10, 1000)]
# parameter_names = ['P-Gate_Oxide_Thickness', 'P-Channel_Length', 'P-Spacer_Width']
# performance_names = ['P-Threshold_Voltage', 'P-Drain_Current', 'P-Output_Resistance']
# num_doe = 50
# output_csv = 'Device_Process.csv' #normalized data
# output_raw_csv = 'Device_Process_raw.csv' #raw data
# output_parameter_png = 'Device_parameters.png'
# output_performance_png = 'Device_performance.png'

# ###Function
# def Process_model(Gate_Oxide_Thickness, Channel_Length, Spacer_Width):
#     Threshold_Voltage = 1.2 - (0.0005 * Gate_Oxide_Thickness) - (0.001 * Channel_Length) + (0.0001 * Spacer_Width)
#     Drain_Current = 1 - (0.0001 * Gate_Oxide_Thickness) + (0.001 * Channel_Length) - (0.0001 * Spacer_Width)
#     Output_Resistance = 10000 - (10 * Gate_Oxide_Thickness) - (1 * Channel_Length) + (0.1 * Spacer_Width)
#     return [Threshold_Voltage, Drain_Current, Output_Resistance]


#############  Example usage for Device2 Simulator

# ###Settings
# parameter_ranges = [(5, 50), (10, 100), (10, 1000)]
# parameter_names = ['N-Gate_Oxide_Thickness', 'N-Channel_Length', 'N-Spacer_Width']
# performance_names = ['N-Threshold_Voltage', 'N-Drain_Current', 'N-Output_Resistance']
# num_doe = 50
# output_csv = 'Device2_Process.csv' #normalized data
# output_raw_csv = 'Device2_Process_raw.csv' #raw data
# output_parameter_png = 'Device2_parameters.png'
# output_performance_png = 'Device2_performance.png'

# ###Function
# def Process_model(Gate_Oxide_Thickness, Channel_Length, Spacer_Width):
#     Threshold_Voltage = 1.2 - (0.0005 * Gate_Oxide_Thickness) - (0.001 * Channel_Length) + (0.0001 * Spacer_Width)
#     Drain_Current = 1 - (0.0001 * Gate_Oxide_Thickness) + (0.001 * Channel_Length) - (0.0001 * Spacer_Width)
#     Output_Resistance = 10000 - (10 * Gate_Oxide_Thickness) - (1 * Channel_Length) + (0.1 * Spacer_Width)
#     return [Threshold_Voltage, Drain_Current, Output_Resistance]

#############  Example usage for SRAM Simulator

###Settings
parameter_ranges = [(100, 10000), (0.5, 1.5), (1, 10)]
parameter_names = ['Cell_Size', 'Precharge_Voltage', 'Sense_Amplifier_Timing']
performance_names = ['Read_Stability', 'Write_Access_Time', 'Power_Consumption']
num_doe = 50
output_csv = 'SRAM_Process.csv' #normalized data
output_raw_csv = 'SRAM_Process_raw.csv' #raw data
output_parameter_png = 'SRAM_parameters.png'
output_performance_png = 'SRAM_performance.png'

###Function
def Process_model(Cell_Size, Precharge_Voltage, Sense_Amplifier_Timing):
    Read_Stability = 100 - (0.001 * Cell_Size) + (2 * Precharge_Voltage) - (0.5 * Sense_Amplifier_Timing)
    Write_Access_Time = 40 - (0.001 * Cell_Size) - (5 * Precharge_Voltage) + (1 * Sense_Amplifier_Timing)
    Power_Consumption = 100 - (0.005 * Cell_Size) + (10 * Precharge_Voltage) + (0.1 * Sense_Amplifier_Timing)
    return [Read_Stability, Write_Access_Time, Power_Consumption]


#############  Example usage for RO

# ###Settings
# parameter_ranges = [(0.5, 3.3), (0.1, 10), (0.1, 1)]
# parameter_names = ['Supply_Voltage', 'Load_Capacitance', 'Inverter_Delay']
# performance_names = ['Frequency', 'Power_Consumption', 'Phase_Noise']
# num_doe = 50
# output_csv = 'RO_Process.csv' #normalized data
# output_raw_csv = 'RO_Process_raw.csv' #raw data
# output_parameter_png = 'RO_parameters.png'
# output_performance_png = 'RO_performance.png'

# ###Function
# def Process_model(Supply_Voltage, Load_Capacitance, Inverter_Delay):
    Frequency = 1e6 * (Supply_Voltage - 0.5) / (Load_Capacitance * Inverter_Delay)
    Power_Consumption = 100 * (Supply_Voltage ** 2) * Load_Capacitance * Frequency
    Phase_Noise = -160 + 40 * Inverter_Delay + 10 * (Load_Capacitance - 0.1) - 10 * (Supply_Voltage - 0.5)
    return [Frequency, Power_Consumption, Phase_Noise]


    
#######################Part 2: Base class Simulator and class GeneralSimulator:

class Simulator(ABC):
    @abstractmethod
    def simulate(self):
        pass

class GeneralSimulator(Simulator):
    def __init__(self, model_func, parameters, performance_names):
        self.model_func = model_func
        self.parameters = parameters
        self.performance_names = performance_names

    def simulate(self):
        return self.model_func(*self.parameters)



##################  Part 3: Operation functions

def generate_doe_parameters(parameter_ranges, n):
    doe_parameters = []
    for _ in range(n):
        parameters = [random.uniform(low, high) for low, high in parameter_ranges]
        doe_parameters.append(parameters)
    return doe_parameters

def write_to_csv(filename, doe_parameters, results, parameter_names, performance_names):
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(parameter_names + performance_names)
        for (params, res) in zip(doe_parameters, results):
            csvwriter.writerow(params + res)

def read_from_csv(filename, parameter_names, performance_names):
    doe_parameters = []
    results = []
    with open(filename, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # skip header
        for row in csvreader:
            values = list(map(float, row))
            doe_parameters.append(values[:len(parameter_names)])
            results.append(values[len(parameter_names):])
    return doe_parameters, results

def normalize_results(filename, item_names):
    
# Load CSV file into a Pandas DataFrame
    df = pd.read_csv(filename)
    
# Initialize the scaler object
    scaler = MinMaxScaler(feature_range=(0, 10))
    
# Normalize the selected columns
    df[item_names] = scaler.fit_transform(df[item_names])

# Write the normalized data back to the CSV file
    df.to_csv(filename, index=False)

# Initialize the scaler object
    scaler = MinMaxScaler(feature_range=(0, 10))

# Normalize the selected columns
    df[item_names] = scaler.fit_transform(df[item_names])

# Write the normalized data back to the CSV file
    df.to_csv(filename, index=False)

#################Part 4: Run and plot

doe_parameters = generate_doe_parameters(parameter_ranges, num_doe)
results = [GeneralSimulator(Process_model, params, performance_names).simulate() for params in doe_parameters]
# print(results)
## log operation for results data
#results = np.log(np.array(results))
results = np.abs(results)
results = np.log(results)
# print(results)
results = results.tolist()  # convert data2 back to a list of lists
for row in results:
    ", ".join(str(elem) for elem in row)
# print(results)

## Write DOE parameters and simulated results to the CSV file

write_to_csv(output_raw_csv, doe_parameters, results, parameter_names, performance_names)
write_to_csv(output_csv, doe_parameters, results, parameter_names, performance_names)

## Normalize results to the CSV file

normalize_results(output_csv, performance_names)
normalize_results(output_csv, parameter_names)

## Read DOE parameters and results from the CSV file

doe_parameters, results = read_from_csv(output_csv, parameter_names, performance_names)

## Combine DOE parameters and results into a single DataFrame

doe_parameters_df = pd.DataFrame(doe_parameters, columns=parameter_names)
results_df = pd.DataFrame(results, columns=performance_names)
data = pd.concat([doe_parameters_df, results_df], axis=1)

## Prepare data for plotting

num_params = doe_parameters_df.shape[1]
num_metrics = results_df.shape[1]
angles = np.linspace(0, 2 * np.pi, num_params, endpoint=False).tolist()


def plot_radar_charts(data, title, highlight_strategies, highlight_colors):
#     fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    ax = plt.subplot(111, projection='polar')
    
    num_vars = len(data.columns)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    def plot_data(row, color, alpha=1, linewidth=2, marker="o"):
        row += row[:1]
        angles_plot = angles + angles[:1]
        ax.plot(angles_plot, row, color=color, alpha=alpha, linewidth=linewidth, marker=marker, markersize=3)    

    for i, row in data.iterrows():
        if i not in highlight_strategies:
            plot_data(row.tolist(), color="gray", alpha=0.5, linewidth=2)

    for i, row in data.iterrows():
        if i in highlight_strategies:
            color_index = highlight_strategies.index(i)
            plot_data(row.tolist(), color=highlight_colors[color_index], alpha=1, linewidth=5)

    r_label_distance = 10
    for angle, label in zip(angles, data.columns):
        x = angle
        y = r_label_distance
        
        if 0 < angle < np.pi:
            ha = 'left'
            va = 'center'
        elif np.pi< angle < 2 * np.pi:
            ha = 'right'
            va = 'center'
        elif angle == 0 * np.pi:
            ha = 'center'
            va = 'bottom'
        else:
            ha = 'center'
            va = 'top'

        ax.annotate(label, xy=(x, y), fontsize=18, fontweight='bold', color='black', ha=ha, va=va)            

    # Customize the chart
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.xaxis.set_tick_params(pad=15)
    ax.set_thetagrids(np.degrees(angles), data.columns, fontsize=1, color='white')
    ax.set_rlabel_position(180 / num_params)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=10, color='black')
    ax.set_title(title, fontweight='bold', fontsize=26, color='black')
 
    return ax

# fig = plt.figure(figsize=(5, 5))

# Plot the parameters and performance metrics radar charts
ax1 = plot_radar_charts(doe_parameters_df, "Parameter Tuning", [0, 1, 2], ['blue', 'red', 'green'])
plt.tight_layout()
plt.savefig(output_parameter_png, dpi=300, bbox_inches='tight')
plt.show()

ax2 = plot_radar_charts(results_df, "Performance Metric", [0, 1, 2], ['blue', 'red', 'green'])
plt.tight_layout()
plt.savefig(output_performance_png, dpi=300, bbox_inches='tight')
plt.show()


