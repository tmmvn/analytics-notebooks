# -*- coding: utf-8 -*-

# -- Sheet --

# # Economic Analysis by Tommi Venemies
# In This workbook, I conduct analyzes on economic world data. I originally planned on scraping data from the CIA World Factbook, which is a great resource for a lot of data. However, CIA has recently updated the site to a more modern outtake, and at the same time have prevented scraping I was testing earlier. Thus, I have scoured the data from elsewhere. Mainly, WorldBank, and IMF.
# 
# The data has been cleaned locally in a spreadsheet application and exported to a .csv. For some reason, many government agencies like to make their spreadsheets hard to analyze straight off the bat.
# 
# The Analysis is conducted on the assumption, that population is actually a key variable for a whole lot of things, and that traditional metrics might hide information that will surface by taking population and demographics into account.
# 
# This is simply a hobby analysis working on a random hypothesis; take it with a grain of salt. While data is a science, the hypothesis and the source of them are personal opinions.
# 
# ## Let's begin
# Let us start by importing data and needed analysis classes.


import pandas as pd
from lets_plot import *
from lets_plot.geo_data import *

df = pd.read_csv('/data/workspace_files/countries.csv', delimiter=';')
countries_df = df.copy()
df = pd.read_csv('/data/workspace_files/gdp.csv', delimiter=';')
gdp_df = df.copy()
df = pd.read_csv('/data/workspace_files/pop.csv', delimiter=';')
pop_df = df.copy()
gdp_df = pd.merge(countries_df, gdp_df, on=['Country Code'])
pop_df = pd.merge(countries_df, pop_df, on=['Country Code'])

countries_geocoder = geocode_countries(gdp_df['Country']) \
    .allow_ambiguous()

# With the data ready, let's first render the world population to a map as that is the basis to the hypothesis. The render shows the hotspots of population. When other things are rendered to similar maps, visual comparison should be easy.
# 
# There are some todo items at this stage:
# - Figure out why some countries are not matched, e.g. Russia is missing though it is in the .csv
# - Think out colors, with some basic color science the visual comparisons could be more meaningful (e.g. red mixed with blue is purple)


ggplot(pop_df) + geom_livemap() + \
        geom_polygon(aes(fill='2019'), map=countries_geocoder, map_join=['Country']) + \
        ggtitle('Population') + \
        scale_fill_gradient(low='white', high="red") + ggsize(760, 300)

