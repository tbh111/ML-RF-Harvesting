#  generate dataset with spice parameters and exported file of ads
import numpy as np
import pandas as pd

spice_data = './spice_data/spice_data.csv'
# ads_export = './spice_data/ads_export_mass_0-10.csv'
ads_export = './spice_data/ads_export_mass_0-10_f=2.3-2.6.csv'
spice_pd = pd.read_csv(spice_data)
ads_pd = pd.read_csv(ads_export)
# ads_pd.columns = ['Iter', 'Eff', 'RL', 'Pin']  # f0=2.45 version
ads_pd.columns = ['Iter', 'Eff', 'RL', 'F0', 'Pin']
diode_total = 11
dataset = pd.DataFrame()
spice_param = ['Bv', 'Cj0', 'Eg','Ibv', 'Is', 'N', 'Rs', 'Vj', 'Xti', 'M', 'Tt']

spice_pd = spice_pd[:diode_total]
# print(spice_pd)
# print(ads_pd.head())

for i in range(0, diode_total):
    print('reading diode '+str(i))
    data_temp = ads_pd.loc[lambda df: df['Iter'] == i, :]
    for j in range(len(spice_param)):
        # data_temp[spice_param[j]] = spice_pd.iloc[i, j+1]
        x = data_temp.copy()  # to remove pandas warning
        x.loc[:, spice_param[j]] = spice_pd.iloc[i, j+1]
        data_temp = x
    data_temp = data_temp.drop(['Iter'], axis=1)  # remove diode iterator
    data_temp[['Pin', 'Eff']] = data_temp[['Eff', 'Pin']]
    dataset = dataset.append(data_temp)
    # print(data_temp)  # 13-dimensional data input, 1-dimensional data output
    print('-----------------')

print(dataset)
dataset.to_csv('./spice_data/spice_reformat_dataset_mass_f=2.3-2.6.csv', header=False, index=False)