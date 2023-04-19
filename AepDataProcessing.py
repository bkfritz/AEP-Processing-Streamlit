# import numpy, pandas, and streamlit
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import io
import base64
import matplotlib as mpl
import warnings
import zipfile

warnings.filterwarnings('ignore')

mpl.rcParams['axes.linewidth'] = 3

# To run: streamlit run AepDataProcessing.py

buffer = io.BytesIO()

def CPDA_Donut(values, ax=None, **plt_kwargs):
    # values are the AEP too small, just right, too big
    if ax is None:
        ax = plt.gca()
    
    if len(values) == 0:
        ax.text(-0.3, -0.2, 'N/A', fontsize=80,  color='black')
    else:
        explode = (0, 0, 0)
        colors = ['lightcoral', 'mediumseagreen', 'royalblue']
        ax.pie(values, explode=explode, colors=colors,
                autopct='', shadow=False, startangle=140, pctdistance = 1.1,
                textprops={'fontsize': 16}, labeldistance=1.2,
                wedgeprops={"edgecolor":"k",'linewidth': 2, 'antialiased': True})

        centre_circle = plt.Circle((0,0),0.65,fc='white',ec='black',lw=2)
        ax.add_patch(centre_circle)

        # Add text for Percent of AEP Just Right
        ax.text(0, 0.18, str(int(values[1]))+'%', fontsize=35,  color='mediumseagreen', 
                        ha='center', va='center')

        # Add text for Percent of AEP Too Small
        ax.text(0, -0.37, str(int(values[0]))+'%', fontsize=30,  color='lightcoral', 
                        ha='center', va='center')

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
    index = next((i for i, name in enumerate(nozzles) if "11006" in name), None)
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
def process_excel_file(uploaded_file):
    sheet_names = pd.read_excel(uploaded_file, sheet_name=None).keys()
    selected_sheet = st.sidebar.selectbox("Select sheet", list(sheet_names))
    # Specify the columns to read in the excel file, include A, B, D, F
    columns_to_read = 'A:N, AE:BI'
    # read the selected sheet
    df = pd.read_excel(uploaded_file, sheet_name=selected_sheet, usecols=columns_to_read)
    # Convert all columns names to string
    df.columns = df.columns.astype(str)
    # Remove whitespace from column names and convert to string
    df.columns = df.columns.str.replace(' ', '')
    return selected_sheet, df

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
def write_excel_file(df, sheet_name):
    st.sidebar.write("Filename defaults to Sheet name selected")
    file_name = st.sidebar.text_input("If desired, enter new filename to save current worksheet AEP results", sheet_name+".xlsx")
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # write means to Excel file
        df.to_excel(writer, sheet_name='Data', index=False)

    st.sidebar.download_button(label='Download Excel Data', data=buffer,
                           file_name=file_name, mime='application/vnd.ms-excel')

def CPDA_Donut(values, ax=None, **plt_kwargs):
    # values are the AEP too small, just right, too big
    if ax is None:
        ax = plt.gca()
    
    if len(values) == 0:
        ax.text(-0.3, -0.2, 'N/A', fontsize=80,  color='black')
    else:
        explode = (0, 0, 0)
        colors = ['royalblue', 'mediumseagreen','lightcoral']
        ax.pie(values, explode=explode, colors=colors,
                autopct='', shadow=False, startangle=140, pctdistance = 1.1,
                textprops={'fontsize': 16}, labeldistance=1.2,
                wedgeprops={"edgecolor":"k",'linewidth': 2, 'antialiased': True})

        centre_circle = plt.Circle((0,0),0.65,fc='white',ec='black',lw=2)
        ax.add_patch(centre_circle)

        # Add text for Percent of AEP Just Right
        ax.text(0, 0.18, str(int(values[1]))+'%', fontsize=35,  color='black', 
                        ha='center', va='center')

        # Add text for Percent of AEP Too Small
        ax.text(0, -0.37, str(int(values[0]))+'%', fontsize=30,  color='red', 
                        ha='center', va='center')

def CPDA_Titles(text, bkgd_color, text_color, ax=None, show_text=True, **tit_kwargs):
    if ax is None:
        ax = plt.gca()
    if show_text == True:
        if text == 'RoundupPowerMax':
            text1 = 'Loaded Cationic\nSoluble Liquid\n(SL)'
            text2 = 'Ex: glyphosate'
            x1, y1 = 0.5, 0.65
            x2, y2 = 0.5, 0.35
            fontsize1 = 25
        elif text == '2,4-DAmine4':
            text1 = 'Loaded Anionic\nSoluble Liquid\n(SL)'
            text2 = 'Ex: glufosinate'
            x1, y1 = 0.5, 0.65
            x2, y2 = 0.5, 0.35
            fontsize1 = 25
        elif text == 'Liberty':
            text1 = 'No-load\nSoluble Liquid\n(SL)'
            text2 = 'Ex: 2, 4-D amine'
            x1, y1 = 0.5, 0.65
            x2, y2 = 0.5, 0.35
            fontsize1 = 25
        elif text == 'Tilt':
            text1 = 'Emulsifiable\nConcentrate\n(EC)'
            text2 = 'Ex: pyraclostrobin'
            x1, y1 = 0.5, 0.65
            x2, y2 = 0.5, 0.35
            fontsize1 = 25
        elif text == 'XR11004':
            text1 = text
            text2 = 'single orifice\nflat fan'
            x1, y1 = 0.5, 0.65
            x2, y2 = 0.5, 0.35
            fontsize1 = 25
        elif text == 'TT11004':
            text1 = text
            text2 = 'turbulence\nchamber'
            x1, y1 = 0.5, 0.65
            x2, y2 = 0.5, 0.35
            fontsize1 = 25
        elif text == 'AIXR11004':
            text1 = text
            text2 = 'air induction\nflat fan'
            x1, y1 = 0.5, 0.65
            x2, y2 = 0.5, 0.35
            fontsize1 = 25
        elif text == 'TTI11004':
            text1 = text
            text2 = 'air inducted\nturbulence\nchamber'
            x1, y1 = 0.5, 0.65
            x2, y2 = 0.5, 0.35
            fontsize1 = 25
        else:
            text1 = text
            text2 = ''
            x1, y1 = 0.5, 0.5
            x2, y2 = 0.5, 0.5
            fontsize1 = 30
        ax.text(x1, y1, text1, fontsize = fontsize1, weight='normal',
                color = text_color, ha='center', va = 'center')
        ax.text(x2, y2, text2, fontsize = 20, weight='normal', color='white', 
                ha='center', va = 'center')
    ax.set_facecolor(bkgd_color)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

def getAdjuvantData(df, adjuvant):
    # Get only data associated with adjuvant
    subdf = df[(df['Adj'] == adjuvant)]
    return subdf

def getActiveNozzleData(df, active, nozzle):
    # Get only the data for the active and nozzle
    subdf = df[(df['Active'] == active) & (df['Nozzle'] == nozzle)]
    # Get Too Small, Too Big, and Just Right AEP values
    subdf = subdf[['Too Small', 'Just Right', 'Too Big']]
    # Subdf to list
    arr = subdf.values.tolist()[0]
    return arr

def addProductData(df):
    # Load AEP results from excel fil aep_data.xlsx into df
    # From Streamlit, this would just be a copy of the processed data df
    df = df.copy()

    # Remove whitespaces from Nozzle column
    df['Nozzle'] = df['Nozzle'].str.strip()
    # Remove whitespaces in Active names
    df['Nozzle'] = df['Nozzle'].str.replace(' ', '')

    # Split solution columns values by "+" and create new column
    df['Active, Adj'] = df['Solution'].str.split('+')

    # Create new columns for the two values in "Active, Adj" column
    df['Active'] = df['Active, Adj'].str[0]
    # Removing the whitespaces in Active fixes any user data entry errors
    # and allows for easier comparison of active names
    # Remove whitespaces from Active column
    df['Active'] = df['Active'].str.strip()
    # Remove whitespaces in Active names
    df['Active'] = df['Active'].str.replace(' ', '')
    # Drop Water
    df = df[df['Active'] != 'Water']

    df['Adj'] = df['Active, Adj'].str[1]

    return df

def createAdjuvantAEPRatingFigure(adj_df):
        # The following are set by the CPDA Program and should  not change
    # All whitespaces removed to hopefully prevent issues due to user entry errors
    nozzles = ['XR11004', 'TT11004', 'AIXR11004', 'TTI11004']
    actives = ['RoundupPowerMax', 'Liberty', '2,4-DAmine4', 'Tilt']

    # Create figure of 5x5 subplots
    fig = plt.figure(figsize=(20,20))
    axes = fig.subplots(5,5)

    # add axes for Adjuvant name
    ax0 = CPDA_Titles('', 'royalblue', 'black', ax=axes[0,0], show_text=False)
    # add subplots that show nozzle names
    ax1 = CPDA_Titles(nozzles[0], 'lightsteelblue', 'black', ax=axes[0,1])
    ax2 = CPDA_Titles(nozzles[1], 'lightsteelblue', 'black', ax=axes[0,2])
    ax3 = CPDA_Titles(nozzles[2], 'lightsteelblue', 'black', ax=axes[0,3])
    ax4 = CPDA_Titles(nozzles[3], 'lightsteelblue', 'black', ax=axes[0,4])

    # Roundup PowerMax data row
    ax5 = CPDA_Titles(actives[0], 'lightsteelblue', 'black', ax=axes[1,0])
    ax6 = CPDA_Donut(getActiveNozzleData(adj_df, actives[0], nozzles[0]), ax=axes[1,1])
    ax7 = CPDA_Donut(getActiveNozzleData(adj_df, actives[0], nozzles[1]), ax=axes[1,2])
    ax8 = CPDA_Donut(getActiveNozzleData(adj_df, actives[0], nozzles[2]), ax=axes[1,3])
    ax9 = CPDA_Donut(getActiveNozzleData(adj_df, actives[0], nozzles[3]), ax=axes[1,4])

    # Liberty data row
    ax10 = CPDA_Titles(actives[1], 'lightsteelblue', 'black', ax=axes[2,0])
    ax11 = CPDA_Donut(getActiveNozzleData(adj_df, actives[1], nozzles[0]), ax=axes[2,1])
    ax12 = CPDA_Donut(getActiveNozzleData(adj_df, actives[1], nozzles[1]), ax=axes[2,2])
    ax13 = CPDA_Donut(getActiveNozzleData(adj_df, actives[1], nozzles[2]), ax=axes[2,3])
    ax14 = CPDA_Donut(getActiveNozzleData(adj_df, actives[1], nozzles[3]), ax=axes[2,4])

    # 2,4-DAmine4 data row
    ax15 = CPDA_Titles(actives[2], 'lightsteelblue', 'black', ax=axes[3,0])
    ax16 = CPDA_Donut(getActiveNozzleData(adj_df, actives[2], nozzles[0]), ax=axes[3,1])
    ax17 = CPDA_Donut(getActiveNozzleData(adj_df, actives[2], nozzles[1]), ax=axes[3,2])
    ax18 = CPDA_Donut(getActiveNozzleData(adj_df, actives[2], nozzles[2]), ax=axes[3,3])
    ax19 = CPDA_Donut(getActiveNozzleData(adj_df, actives[2], nozzles[3]), ax=axes[3,4])

    # Tilt data row
    ax20 = CPDA_Titles(actives[3], 'lightsteelblue', 'black', ax=axes[4,0])
    ax21 = CPDA_Donut(getActiveNozzleData(adj_df, actives[3], nozzles[0]), ax=axes[4,1])
    ax22 = CPDA_Donut(getActiveNozzleData(adj_df, actives[3], nozzles[1]), ax=axes[4,2])
    ax23 = CPDA_Donut(getActiveNozzleData(adj_df, actives[3], nozzles[2]), ax=axes[4,3])
    ax24 = CPDA_Donut(getActiveNozzleData(adj_df, actives[3], nozzles[3]), ax=axes[4,4])

    fig.subplots_adjust(wspace=0, hspace=0)

    return fig

def plot_figures(df, adjuvants):
    figs = []
    for adj in adjuvants:
        # Filter data by selected adjuvant
        filtered_data = df[df['Adj'] == adj]
        fig = createAdjuvantAEPRatingFigure(filtered_data)
        figs.append(fig)
    return figs

def get_zip_file(figs, df):
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, mode='w') as archive:
        for i, fig in enumerate(figs):
            png_buffer = io.BytesIO()
            fig.savefig(png_buffer, format='png')
            png_buffer.seek(0)
            archive.writestr(f'figure_{i}.png', png_buffer.read())

        # Save the dataframe to Excel
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, engine='xlswriter', index=False)
        excel_buffer.seek(0)
        archive.writestr('data.xlsx', excel_buffer.read())
    
    zip_buffer.seek(0)
    return zip_buffer

# Main Streamlit app
def main():
    st.sidebar.title("Droplet Size Parameters")
    st.sidebar.write('''
    ## About
    

    This app assumes that the Excel file contains the following columns in the following order in columns A:N:


    Date, Time, Range, Solution, Nozzle, Nozzle Orifice, Pressure, Airspeed, Rep, Dv10, Dv50, Dv90, RS


    Followed by the 31 incremental distribution columns: AE through BI

    
    It is also assumed that the reference nozzle data names contain the strings:


    11001, 11003, 11006, 8008, 6510, and 6515.


    If these assumptions are not met, the app will not work correctly.


    If the excel file contains more than one worksheet of data, the user will have to selected each worksheet individually and save the resulting data to separate Excel files.
    
    
    ''')

    st.sidebar.write("Upload an Excel file with droplet size data")
    # Read Excel file
    uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx", "xls"])
    if uploaded_file:
        sheet_name, df = process_excel_file(uploaded_file)
        if df is not None:
            st.write("Calculating means for each unique combination of nozzle and solution...")
            means = calculate_means(df)
            # write too_small value to screen
            st.write("Too small cutoff value (11003 Dv10):")    
            st.write(round(get_too_small(means),0))
            # write too_big value to screen
            st.write("Too big cutoff value (average of 8008 and 6510 Dv90):")
            st.write(round(get_too_big(means),0))
            aep_fracs = calc_aep_fractions(means)
            st.write("All AEP Results for Adjuvants in Current Worksheet:")
            st.write(aep_fracs)

            # # Add Button to download means to Excel file
            # write_excel_file(aep_fracs, sheet_name)

            aep_fracs = addProductData(aep_fracs)
            # Get list of unique adjuvants
            adj_list = aep_fracs['Adj'].unique()
            # Drop nan values
            adj_list = adj_list[~pd.isnull(adj_list)]

            # # Show selectbox with adjuvants
            # selected_adjuvant = st.sidebar.selectbox('Select adjuvant name to see AEP only for that product:', adj_list)

            # # Filter data by selected adjuvant
            # filtered_data = df[df['Adj'] == selected_adjuvant]

            # # Display filtered data
            # st.write("AEP Results for Product: " + selected_adjuvant + " only")
            # st.write(filtered_data)

            # Generate and display the figures
            figs = plot_figures(aep_fracs, adj_list)
            for fig in figs:
                st.write(fig)
            
            # Download figures as images

            # Add Button to download means to Excel file, show and download plots
            # write_excel_file(aep_fracs, sheet_name, figs)
            if st.sidebar.button('Download Excel File and Plots'):
                zip_buffer = get_zip_file(figs, aep_fracs)
                zip_base64 = base64.b64encode(zip_buffer.read()).decode('ascii')
                href = f'<a href="data:application/zip;base64,{zip_base64}" download="plots.zip">Download ZIP</a>'
                st.markdown(href, unsafe_allow_html=True)

                # with open(zip_filename, 'rb') as f:
                #     zip_base64 = base64.b64encode(f.read()).decode('ascii')
                #     href = f'<a href="data:application/zip;base64,{zip_base64}" download=\'{zip_filename}\'>Download ZIP</a>'
                #     st.markdown(href, unsafe_allow_html=True)

                # # Iterate through adjuvants
                # for adjuvant in adj_list:

                #     # Filter data by adjuvant
                #     filtered_data = getAdjuvantData(df, adjuvant)

                #     # Create plot
                #     fig = createAdjuvantAEPRatingFigure(filtered_data)
                #     plots.append(fig)

                #     st.write(f'Results Plot for {adjuvant}:')
                #     st.pyplot(fig)

                #     # Save plot
                #     fig.savefig('AEP Results Table for ' + adjuvant + '.png', dpi=300)

if __name__ == '__main__':
    main()