import glob
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio

def read_multiple_csv_files(*file_names):
    data_frames = []
    
    for file in file_names:
        data_frame = pd.read_csv(file)
        data_frames.append(data_frame)

    return pd.concat(data_frames, ignore_index=True)

data = read_multiple_csv_files('Combined-2.csv')

# Highlight specific rows (customize this as needed)
highlighted_rows = [49, 0]  # Indices of the rows to highlight
highlighted_colors = ['red', 'blue']  # Custom colors for highlighted rows
highlighted_line_width = 4  # Width of highlighted lines

# Create a list of colors and line widths for each row
colors = ['gray' if i not in highlighted_rows else highlighted_colors[highlighted_rows.index(i)] for i in range(len(data))]
line_widths = [1 if i not in highlighted_rows else highlighted_line_width for i in range(len(data))]

# Create a pseudo parallel coordinate plot using scatter traces
fig = go.Figure()

# Loop through each column
for i in range(len(data.columns) - 1):
    # Draw non-highlighted lines first
    for j, row in data.iterrows():
        if j in highlighted_rows:
            continue
        fig.add_trace(go.Scatter(
            x=[i, i + 1],
            y=[row[data.columns[i]], row[data.columns[i + 1]]],
            mode='lines+markers',
            marker=dict(size=3),
            line=dict(color=colors[j], width=line_widths[j]),
            showlegend=False,
            yaxis='y2'
        ))

# Loop through each column again for highlighted lines
for i in range(len(data.columns) - 1):
    # Draw highlighted lines on top
    for j, row in data.iterrows():
        if j not in highlighted_rows:
            continue
        fig.add_trace(go.Scatter(
            x=[i, i + 1],
            y=[row[data.columns[i]], row[data.columns[i + 1]]],
            mode='lines+markers',
            marker=dict(size=1),
            line=dict(color=colors[j], width=line_widths[j]),
            showlegend=False,
            yaxis='y2'
        ))

# # Add scatter traces for each column
# for i in range(len(data.columns) - 1):
#     # Draw non-highlighted lines first
#     for j, row in data.iterrows():
#         if j in highlighted_rows:
#             continue
#         fig.add_trace(go.Scatter(
#             x=[i, i + 1],
#             y=[row[data.columns[i]], row[data.columns[i + 1]]],
#             mode='lines+markers',
#             marker=dict(size=6),
#             line=dict(color=colors[j], width=line_widths[j]),
#             showlegend=False,
#             yaxis='y2'
#         ))
#     # Draw highlighted lines on top
#     for j, row in data.iterrows():
#         if j not in highlighted_rows:
#             continue
#         fig.add_trace(go.Scatter(
#             x=[i, i + 1],
#             y=[row[data.columns[i]], row[data.columns[i + 1]]],
#             mode='lines+markers',
#             marker=dict(size=6),
#             line=dict(color=colors[j], width=line_widths[j]),
#             showlegend=False,
#             yaxis='y2'
#         ))

# Customize the plot appearance
fig.update_xaxes(tickvals=list(range(len(data.columns))), ticktext=data.columns, tickangle=-65, tickfont=dict(size=28))
fig.update_yaxes(tickvals=[], showticklabels=False)

fig.update_layout(
    font=dict(size=32),
    plot_bgcolor='white',
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False),
    hovermode=False,
)

# Add left y-axis with specified labels and horizontal lines
fig.update_layout(
    yaxis2=dict(
        tickvals=[0, 2, 4, 6, 8, 10],
        tickmode='array',
        anchor='free',
        overlaying='y',
        side='left',
        position=0,
        showgrid=True,
        gridcolor='lightgray',
        gridwidth=1,
        title='Normalized(a.u.)',
        zeroline=False,
    )
)

width = 4000  # pixels
height = 800  # pixels

desired_dpi = 300
default_dpi = 96  # The default DPI for Plotly images
scale = desired_dpi / default_dpi
pio.write_image(fig, 'Parallel-4.png', width=width, height=height, scale=scale)

fig.show()
