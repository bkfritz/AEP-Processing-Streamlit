import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
import matplotlib as mpl
import warnings
warnings.filterwarnings('ignore')

mpl.rcParams['axes.linewidth'] = 3

def CPDA_Donut(values, ax=None, **plt_kwargs):
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
    
    #ax.axis('equal')
    #draw circle
        centre_circle = plt.Circle((0,0),0.65,fc='white',ec='black',lw=2)
        ax.add_patch(centre_circle)
        ax.text(-0.3, -0.2, str(int(values[1])), fontsize=50,  color='black')

def CPDA_Titles(text, bkgd_color, text_color, ax=None, **tit_kwargs):
    if ax is None:
        ax = plt.gca()

    ax.text(0.5,0.5, text, fontsize = 30, weight='normal',
                color = text_color, ha='center', va = 'center')
    ax.set_facecolor(bkgd_color)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    
def Get_Data(df, act, adj, Noz):

    solns = act + '+' + adj
    
    Sub_df = df.loc[ (df['Solution'] == solns) & (df['Nozzle'] == Noz)]
    if Sub_df.empty == True:
        Size_Data = 0
    else:
        Size_Data = [Sub_df.iloc[0]['Big'], Sub_df.iloc[0]['Eff'], Sub_df.iloc[0][150]]
    return Size_Data


fig = plt.figure(figsize=(20,20))
axes = fig.subplots(5,5)

Main = Adjuvant
ax0 = CPDA_Titles(Main, 'royalblue','white', ax=axes[0,0])


# Gridded plots, with each main gridded plot being for a given adjuvant



Active1 = Actives[0]
Active2 = Actives[1]
ActivePM = Actives[1].split()
Active22 = ActivePM[0] + '\n' + ActivePM[1]
Active3 = Actives[2]
Active4 = Actives[3]

formulation1 = Active1
formulation2 = Active22
formulation3 = Active3
formulation4 = Active4

axf1 = CPDA_Titles(formulation2, 'lightsteelblue','black', ax=axes[1,0])
axf2 = CPDA_Titles(formulation1, 'lightsteelblue','black', ax=axes[2,0])
axf3 = CPDA_Titles(formulation3, 'lightsteelblue','black', ax=axes[3,0])
axf4 = CPDA_Titles(formulation4, 'lightsteelblue','black', ax=axes[4,0])

axN1 = CPDA_Titles(Nozzles[9], 'lightsteelblue','black',ax=axes[0,1])
axN2 = CPDA_Titles(Nozzles[7], 'lightsteelblue','black',ax=axes[0,2])
axN3 = CPDA_Titles(Nozzles[6], 'lightsteelblue','black',ax=axes[0,3])
axN4 = CPDA_Titles(Nozzles[8], 'lightsteelblue','black',ax=axes[0,4])
         
# Nozzle 1 Data
# Noz1S1 = Data.loc[ (Data['Solution'] == Active1 + '+' + Adjuvant) & (Data['Nozzle'] == Nozzles[6])]
# Noz1S2 = Data.loc[ (Data['Solution'] == Active2 + '+' + Adjuvant) & (Data['Nozzle'] == Nozzles[6])]
# Noz1S3 = Data.loc[ (Data['Solution'] == Active3 + '+' + Adjuvant) & (Data['Nozzle'] == Nozzles[6])]
# Noz1S4 = Data.loc[ (Data['Solution'] == Active4 + '+' + Adjuvant) & (Data['Nozzle'] == Nozzles[6])]

# sizeN1S1 = [Noz1S1.iloc[0]['Big'], Noz1S1.iloc[0]['Eff'], Noz1S1.iloc[0][150]]
# sizeN1S2 = [Noz1S2.iloc[0]['Big'], Noz1S2.iloc[0]['Eff'], Noz1S2.iloc[0][150]]
# sizeN1S3 = [Noz1S3.iloc[0]['Big'], Noz1S3.iloc[0]['Eff'], Noz1S3.iloc[0][150]]
# sizeN1S4 = [Noz1S4.iloc[0]['Big'], Noz1S4.iloc[0]['Eff'], Noz1S4.iloc[0][150]]         


N1S1 = Get_Data(Data, Actives[1], Adjuvant, Nozzles[9])
N1S2 = Get_Data(Data, Actives[0], Adjuvant, Nozzles[9])
N1S3 = Get_Data(Data, Actives[2], Adjuvant, Nozzles[9])
N1S4 = Get_Data(Data, Actives[3], Adjuvant, Nozzles[9])

N2S1 = Get_Data(Data, Actives[1], Adjuvant, Nozzles[7])
N2S2 = Get_Data(Data, Actives[0], Adjuvant, Nozzles[7])
N2S3 = Get_Data(Data, Actives[2], Adjuvant, Nozzles[7])
N2S4 = Get_Data(Data, Actives[3], Adjuvant, Nozzles[7])

N3S1 = Get_Data(Data, Actives[1], Adjuvant, Nozzles[6])
N3S2 = Get_Data(Data, Actives[0], Adjuvant, Nozzles[6])
N3S3 = Get_Data(Data, Actives[2], Adjuvant, Nozzles[6])
N3S4 = Get_Data(Data, Actives[3], Adjuvant, Nozzles[6])

N4S1 = Get_Data(Data, Actives[1], Adjuvant, Nozzles[8])
N4S2 = Get_Data(Data, Actives[0], Adjuvant, Nozzles[8])
N4S3 = Get_Data(Data, Actives[2], Adjuvant, Nozzles[8])
N4S4 = Get_Data(Data, Actives[3], Adjuvant, Nozzles[8])


axN1S1 = CPDA_Donut(N1S1, ax=axes[1,1])
axN1S2 = CPDA_Donut(N1S2, ax=axes[2,1])
axN1S3 = CPDA_Donut(N1S3, ax=axes[3,1])
axN1S4 = CPDA_Donut(N1S4, ax=axes[4,1])

axN2S1 = CPDA_Donut(N2S1, ax=axes[1,2])
axN2S2 = CPDA_Donut(N2S2, ax=axes[2,2])
axN2S3 = CPDA_Donut(N2S3, ax=axes[3,2])
axN2S4 = CPDA_Donut(N2S4, ax=axes[4,2])

axN3S1 = CPDA_Donut(N3S1, ax=axes[1,3])
axN3S2 = CPDA_Donut(N3S2, ax=axes[2,3])
axN3S3 = CPDA_Donut(N3S3, ax=axes[3,3])
axN3S4 = CPDA_Donut(N3S4, ax=axes[4,3])

axN4S1 = CPDA_Donut(N4S1, ax=axes[1,4])
axN4S2 = CPDA_Donut(N4S2, ax=axes[2,4])
axN4S3 = CPDA_Donut(N4S3, ax=axes[3,4])
axN4S4 = CPDA_Donut(N4S4, ax=axes[4,4])


fig.subplots_adjust(wspace=0, hspace=0)
plt.show()

name = Adjuvant +'_donut.png'
fig.savefig(name, format='png',bbox_inches='tight', dpi=1000)
