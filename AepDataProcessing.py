# import numpy, pandas, and streamlit
import numpy as np
import pandas as pd
import streamlit as st
import io

# To run: streamlit run AepDataProcessing.py

buffer = io.BytesIO()

def interpolate_volume_fraction(row, diameter, range="Small"):
    diameter = int(diameter)
    # Find columns with diameters that bracket the given diameter
    column_names = list(row.index)
    smaller_diameter = max([d for d in column_names if d < diameter], default=None)
    larger_diameter = min([d for d in column_names if d > diameter], default=None)

    # Check if the given diameter is equal to a column name
    if diameter in column_names:
        volume_fraction = row[diameter]
    elif smaller_diameter is not None and larger_diameter is not None:
        # Interpolate volume fraction between two columns
        x = np.array([float(smaller_diameter), float(larger_diameter)])
        y = np.array([row[int(smaller_diameter)], row[int(larger_diameter)]])
        volume_fraction = np.interp(diameter, x, y)
    else:
        # Diameter is outside the range of the columns
        volume_fraction = np.nan

    if range == "Small":
        volume_fraction = volume_fraction
    else:
        volume_fraction = 100 - volume_fraction
    return volume_fraction

def get_too_small(df):
    # get unique list of nozzle names
    nozzles = df['Nozzle'].unique()
    # Find index of nozzle name that contains "11003"
    index = next((i for i, name in enumerate(nozzles) if "11003" in name), None)
    # if index is not None: set too_small as the Dv10 value using the nozzle name in nozzle at the index
    if index is not None:
        # get Dv10 value from means for nozzle in nozzles at index
        too_small = df.loc[df['Nozzle'] == nozzles[index], 'Dv10'].values[0]
    return too_small

def get_too_big(df):
    # get unique list of nozzle names
    nozzles = df['Nozzle'].unique()
    # Find index of nozzle name that contains "8008"
    index1 = next((i for i, name in enumerate(nozzles) if "8008" in name), None)
    # Find index of nozzle name that contains "6510"
    index2 = next((i for i, name in enumerate(nozzles) if "6510" in name), None)
    # if index1 is not None: set too_big1 as the Dv90 value using the nozzle name in nozzle at the index1
    if index1 is not None:
        too_big1 = df.loc[df['Nozzle'] == nozzles[index1], 'Dv90'].values[0]
    # if index2 is not None: set too_big2 as the Dv90 value using the nozzle name in nozzle at the index2
    if index2 is not None:
        too_big2 = df.loc[df['Nozzle'] == nozzles[index2], 'Dv90'].values[0]
    # Set too_big as the average of too_big1 and too_big2
    too_big = (too_big1 + too_big2) / 2
    return too_big

# Define function to read Excel file and prompt user to select sheet
def read_excel_file():
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
    if uploaded_file:
        sheet_names = pd.read_excel(uploaded_file, sheet_name=None).keys()
        selected_sheet = st.selectbox("Select sheet", list(sheet_names))
        # print selcted sheet to screen
        st.write(f"Selected sheet: {selected_sheet}")
        # Specify the columns to read in the excel file, include A, B, D, F
        columns_to_read = 'A:N, AE:BI'
        # read the selected sheet
        df = pd.read_excel(uploaded_file, sheet_name=selected_sheet, usecols=columns_to_read)
        # Convert all columns names to string
        df.columns = df.columns.astype(str)
        # Remove whitespace from column names and convert to string
        df.columns = df.columns.str.replace(' ', '')
        return df

# Define function to calculate means for each unique combination of nozzle and solution
def calculate_means(df):
    # Calculate means for each unique combination of nozzle and solution
    means = df.groupby(['Nozzle', 'Solution'], as_index=False).mean(numeric_only=True)
    return means

# Define function the gets too_small and calculates the volume fraction for each row
def calc_aep_fractions(df):
    too_small = get_too_small(df)
    too_big = get_too_big(df)
    # Create new data that contains only the last 31 columns from means called incdist, do not add index
    incdist = df.iloc[:, -31:].copy()

    # Convert column names to strings and modify
    incdist.columns = incdist.columns.map(lambda x: int(float(x)))

    # Apply interpolate_volume_fraction function to each row of incdist
    df['Too Small'] = incdist.apply(lambda row: interpolate_volume_fraction(row, too_small, range="Small"), axis=1)
    df['Too Big'] = incdist.apply(lambda row: interpolate_volume_fraction(row, too_big, range="Large"), axis=1)
    df['Just Right'] = 100 - df['Too Small'] - df['Too Big']
    
    return df


# Define function to write mean data to a new Excel file
def write_excel_file(means):
    file_name = st.text_input("Enter file name for mean data", "means.xlsx")
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # write means to Excel file
        means.to_excel(writer, sheet_name='Data', index=False)
        # Close the Pandas Excel writer and output the Excel file.
        writer.save()

        st.download_button(label='Download Excel Data', data=buffer,
                           file_name=file_name, mime='application/vnd.ms-excel')
    # if st.button("Download mean data"):
    #     means.to_excel(file_name, index=False)
    #     st.success(f"Mean data written to {file_name}!")

# Define function the converts dataframe to csv file
def convert_to_csv(df):
    csv = df.to_csv(index=False).encode('utf-8')
    return csv

# Main Streamlit app
def main():
    st.title("Droplet Size Parameters")
    st.write("Upload an Excel file with droplet size data")
    df = read_excel_file()
    if df is not None:
        st.write("Calculating means for each unique combination of nozzle and solution...")
        means = calculate_means(df)
        # write too_small value to screen
        st.write("Too small value:")    
        st.write(round(get_too_small(means),0))
        # write too_big value to screen
        st.write("Too big value:")
        st.write(round(get_too_big(means),0))
        aep_fracs = calc_aep_fractions(means)
        st.write("AEP data:")
        st.write(aep_fracs)

        # Add Button to download means to Excel file
        write_excel_file(aep_fracs)
        
        # Convert aep_fracs df to csv file
        csv = convert_to_csv(aep_fracs)
        # Get filename from user
        filename = st.text_input("Enter filename for AEP data", "")
        # Add Button to download csv file
        st.download_button('Press to Download', csv, filename+'.csv', key='download-csv')

if __name__ == '__main__':
    main()