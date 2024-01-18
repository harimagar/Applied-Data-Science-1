import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cluster_tools as ct
import errors as err
import importlib
import scipy.optimize as opt
from sklearn.cluster import KMeans


def readFile(a,n):
    df=pd.read_csv(a,skiprows=n)
    return df
data=readFile('eco.csv',4)
df_green=data[data['Indicator Name']=='Other greenhouse gas emissions, HFC, PFC and SF6 (thousand metric tons of CO2 equivalent)']
df_fuel=data[data['Indicator Name']=='CO2 emissions from liquid fuel consumption (kt)']

def correlation(df):
    df_green=df.set_index('Country Name', drop=True)
    df_green_cor=df_green.loc[:, '1990':'2000']
    matrix=(df_green_cor).corr()
    sns.heatmap(matrix, annot=True, cmap='vlag')
    plt.show()
    
def linePlot(df):
    df_green=df.set_index('Country Name', drop=True)
    df_green=df_green.loc[:, '1970':'2010']
    df_green_T=df_green.transpose()

    x_values = list(df_green.columns)
    angola_values = df_green_T["Angola"].to_list()
    Albania_value=df_green_T['Japan'].to_list()
    arab_value=df_green_T['Arab World'].to_list()
    # Plotting the data
    plt.plot(x_values, angola_values, linestyle='-',label='Angola')
    plt.plot(x_values, Albania_value, linestyle='--',label='Japan')
    plt.plot(x_values, arab_value, linestyle='-',label='Arab')
    ticks = np.linspace(0, len(list(df_green.columns)) - 1, 5, dtype=int)
    plt.xticks(ticks)
    # plt.xticklabels(list(df_green.columns).iloc[ticks], rotation=20)
    # Adding labels and title
    plt.xlabel('Year')
    plt.ylabel('emission rate')
    plt.title('Other greenhouse gas emissions, HFC, PFC\n& SF6 (thousand metric tons of CO2 equivalent)')
    plt.legend()
    # Display the plot
    plt.show()
    
def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 
    and growth rate g"""
    f = n0 / (1 + np.exp(-g*(t - t0)))

    return f
def forecast_energy(data, country, start_year, end_year):
    data = data.loc[:, country]
    data = data.dropna(axis=0)

    energy = pd.DataFrame()

    energy['Year'] = pd.DataFrame(data.index)
    energy['Energy'] = pd.DataFrame(data.values)
    energy["Year"] = pd.to_numeric(energy["Year"])
    importlib.reload(opt)

    param, covar = opt.curve_fit(logistic, energy["Year"], energy["Energy"],
                                 p0=(1.2e12, 0.03, 1990.0))

    sigma = np.sqrt(np.diag(covar))

    year = np.arange(start_year, end_year)
    forecast = logistic(year, *param)
    error_range = err.error_prop(year, logistic, param, sigma)
    # Assuming the function returns 1D array like: [mean, sigma]
    mean = error_range.iloc[0]
    up = mean + error_range.iloc[1]
    low = mean - error_range.iloc[1]
    plt.figure()
    plt.plot(energy["Year"], energy["Energy"], label="Energy Use")
    plt.plot(year, forecast, label="Forecast", color='k')
    plt.fill_between(year, low, up, color="yellow", alpha=0.7,
                     label='Confidence Margin')
    plt.xlabel("Year")
    plt.ylabel("Energy Use in Kg Oil equivalent per capita")
    plt.legend()
    plt.title(f'Energy Use in Kg Oil equivalent forecast for {country}')
    plt.savefig(f'{country}.png', bbox_inches='tight', dpi=300)
    plt.show()

    energy2030 = logistic(2030, *param)/1e9

    low, up = err.error_prop(2030, logistic, param, sigma)
    sig = np.abs(up-low)/(2.0 * 1e9)
    print()
    print(f"Energy Use in Kg Oil by 2030 in {country}",
          np.round(energy2030*1e9, 2), "+/-", np.round(sig*1e9, 2))

def regplot(data,loc):
    df_fuel=data.set_index('Country Name', drop=True)
    df_fuel=df_fuel.loc[:,'1960':'2010']
    df_fuel_T=df_fuel.transpose()
    Years=list(df_fuel.columns)
    Years = [int(year) for year in Years]
    a_value= np.array(df_fuel_T[f'{loc}'].to_list())
    sns.regplot(x=Years, y=a_value, line_kws={"color": "black"},order=3,ci=90,scatter_kws={'s':50,'alpha':0.9})
    # Set plot title, x-axis label, and y-axis label
    plt.title(f"Increment of CO2 emission from fuel")
    plt.xlabel("Years")
    plt.ylabel("Rate")
    plt.show()

def cluster(df):
    # Choose the number of clusters
    num_clusters = 3
    df_green=df.set_index('Country Name', drop=True)
    df_green=df_green.loc[:, '1970':'2010']
    df_green_T=df_green.transpose()
    # Extract the relevant columns for clustering
    X = df_green_T.loc[:,'Africa Eastern and Southern':'Albania']

    # Create a KMeans model with the specified number of clusters
    kmeans = KMeans(n_clusters=num_clusters)

    # Fit the model to the data
    kmeans.fit(X)

    # Get the cluster labels and centroids
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Plot the data points with different colors for each cluster
    plt.scatter(X['Africa Eastern and Southern'], X['Afghanistan'], c=labels, cmap='viridis', edgecolors='k')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='o', s=50, label='Centroids')
    plt.title('KMeans Clustering for green house\ngas emission')
    plt.legend()
    plt.show()
regplot(df_fuel,'Arab World')
regplot(df_fuel,'Africa Eastern and Southern')
correlation(df_green)
linePlot(df_green)
cluster(df_green)