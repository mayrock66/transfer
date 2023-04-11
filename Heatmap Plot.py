import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def create_heatmap(df, x_col, y_col, z_col, interpolation=False, z_min=None, z_max=None):
    plt.figure(figsize=(10, 8))
    
    if interpolation:
        # Create a grid of points for interpolation
        xi, yi = np.linspace(df[x_col].min(), df[x_col].max(), 500), np.linspace(df[y_col].min(), df[y_col].max(), 500)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate the data
        zi = griddata((df[x_col], df[y_col]), df[z_col], (xi, yi), method='cubic')

        # Create the heatmap using interpolated data
        plt.pcolormesh(xi, yi, zi, cmap='jet', shading='auto', vmin=z_min, vmax=z_max)
    else:
        # Create the heatmap without interpolation
        plt.hist2d(df[x_col], df[y_col], weights=df[z_col], bins=(20, 20), cmap='jet', range=[[df[x_col].min(), df[x_col].max()], [df[y_col].min(), df[y_col].max()]], vmin=z_min, vmax=z_max)
                                                                                                                                   
    plt.xlabel(x_col, fontsize=36, fontweight='bold', color='black')
    plt.ylabel(y_col, fontsize=36, fontweight='bold', color='black')
    cb = plt.colorbar(label=z_col, format="%.1f")
    cb.ax.tick_params(labelsize=24, colors='black')
    cb.set_label(z_col, fontsize=36, fontweight='bold', color='black')
#     cb.set_clim(vmin=0, vmax=10)

    plt.xticks(fontsize=24, color='black')
    plt.yticks(fontsize=24, color='black')
    
#     plt.title(f"{z_col} vs {x_col} and {y_col}", fontsize=48, fontweight='bold', color='black')
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('Heatmap-4', dpi=300, bbox_inches='tight')
    plt.show()

# Example usage
data = pd.read_csv('combined-2-backup.csv')

#Create the heatmap
# x_data = 'Ballistic_efficency'
# y_data = 'Vstat0'
# z_data = 'Electron_conc'

# Create the heatmap
# x_data = 'CD'
# y_data = 'Pad_Hardness'
# z_data = 'Frequency'

# x_data = 'Etch_Selectivity'
# y_data = 'Platen_Speed'
# z_data = 'Surface_Roughness'

x_data = 'Spin_Speed'
y_data = 'Develop_Time'
z_data = 'LER'

create_heatmap(data, x_data, y_data, z_data, interpolation=True)
                                                                                                                                   
                                                                                                                                   # def create_heatmap(df, x_col, y_col, z_col, interpolation=False):
#     plt.figure(figsize=(10, 8))
    
#     if interpolation:
#         # Create a grid of points for interpolation
#         xi, yi = np.linspace(df[x_col].min(), df[x_col].max(), 500), np.linspace(df[y_col].min(), df[y_col].max(), 500)
#         xi, yi = np.meshgrid(xi, yi)

#         # Interpolate the data
#         zi = griddata((df[x_col], df[y_col]), df[z_col], (xi, yi), method='cubic')

#         # Create the heatmap using interpolated data
#         plt.pcolormesh(xi, yi, zi, cmap='jet', shading='auto')
#     else:
#         # Create the heatmap without interpolation
#         plt.hist2d(df[x_col], df[y_col], weights=df[z_col], bins=(20, 20), cmap='jet')
