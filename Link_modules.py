import pandas as pd

# Read CSV data
filename = "combined-2.csv"
data = pd.read_csv(filename)

# Apply physics-based function

## Litho track link to Etch
data['Etch_Uniformity'] = 1 - 0.1 * data['LER'] + 0.1 * data['Resolution'] + 0.8 * data['Etch_Uniformity']
data['CD'] = 2 - 0.1 * data['LER'] - 0.1 * data['Resolution'] + 0.8 * data['CD']

## Etch link to CMP
data['Surface_Roughness'] = 1 - 0.1 * data['Etch_Uniformity'] + 0.1 * data['CD'] + 0.8 * data['Surface_Roughness']
data['Defect_Density'] = 2 - 0.1 * data['CD'] - 0.1 * data['Etch_Uniformity'] + 0.8 * data['Defect_Density']

## CMP link to Device
data['P-Gate_Oxide_THK'] = 2 - 0.1 * data['Removal_Rate'] - 0.1 * data['Polishing_Time'] + 0.8 * data['P-Gate_Oxide_THK']
data['P-Spacer_Width'] = 2 - 0.1 * data['Removal_Rate'] - 0.1 * data['Polishing_Time'] + 0.8 * data['P-Spacer_Width']

## Device link to RO
data['Frequency'] = 1 - 0.1 * data['P-Vt'] + 0.1 * data['P-IdSat'] + 0.8 * data['Frequency']
data['Power_Consumption.2'] = 1 + 0.1 * data['P-Vt'] - 0.1 * data['P-IdSat'] + 0.8 * data['Power_Consumption.2']

## Process link to PMOS Model
data['Ballistic_efficency'] = 2 - 0.1 * data['Defect_Density'] - 0.1 * data['LER'] + 0.8 * data['Ballistic_efficency']
data['Vsat0'] = 1 + 0.1 * data['Defect_Density'] + 0.1 * data['LER'] + 0.8 * data['Vstat0']

## PMOS Model link to PMOS
data['P-Vt'] = 1 + 0.1 * data['Ballistic_efficency'] - 0.1 * data['Vstat0'] + 0.8 * data['P-Vt']
data['P-IdSat'] = 1 - 0.1 * data['Ballistic_efficency'] + 0.1 * data['Vstat0'] + 0.8 * data['P-IdSat']

# Save the results in a new CSV file
new_filename = "Link.csv"
data.to_csv(new_filename, index=False)
