import pandas as pd
import glob
import os

def join_csv_files(input_folder, output_file):
    # Find all CSV files in the input folder
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

    # Exclude the output file from the list of CSV files to be processed
    csv_files = [csv_file for csv_file in csv_files if csv_file != output_file]

    # Read the CSV files and store them in a list of DataFrames
    dataframes = [pd.read_csv(csv_file) for csv_file in csv_files]

    # Concatenate the DataFrames horizontally
    combined_csv = pd.concat(dataframes, axis=1)

    # Write the combined DataFrame to a new CSV file
    combined_csv.to_csv(output_file, index=False, encoding='utf-8-sig')

    # Display the number of rows and columns in the combined CSV file
    print(f"The combined CSV file has {combined_csv.shape[0]} rows and {combined_csv.shape[1]} columns.")

# Example usage
input_folder = "/Users/jack/Documents/Project/Python/TPB demo/Round-8-v3/NA"
output_file = "/Users/jack/Documents/Project/Python/TPB demo/Round-8-v3/combined-2.csv"
join_csv_files(input_folder, output_file)
