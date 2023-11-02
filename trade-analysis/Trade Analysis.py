# -*- coding: utf-8 -*-

# -- Sheet --

# # Trade Analysis by Tommi Venemies
# In this workbook, I conduct analyzes on world trade data. Initially, I will focus on trade from an American perspective instead of global trade flows.
# 
# The data has been cleaned locally in a spreadsheet application and exported to a .csv. For some reason, many government agencies like to make their spreadsheets hard to analyze straight off the bat.
# 
# The analysis is conducted on the assumption, that when trade is adjusted to different variables, the perceived unfairness in trade, in other words trade deficit, changes. The trade deficit seemes to sometimes be a key point in public discourse, stating that a trade deficit means an unfairness towards a nation. However, not all countries are created equal. Thus, it would be sensible to adjust the data to be per capita, or per workforce, or similar, to correct this perception flaw. Afterall, a country with hundred million people has a much larger capacity for buying foreign goods than a country with only a few million people. At least in theory. 
# 
# This is simply a hobbyist analysis working on a random hypothesis; take it with a grain of salt. While data is a science, the hypothesis and the source of them are personal opinions.
# 
# ## Let's begin
# Let us start by importing the data and the needed analysis classes.


import pandas as pd
from lets_plot import *
from lets_plot.geo_data import *

df = pd.read_csv('/data/workspace_files/countries.csv', decimal=",", delimiter=';')
countries_df = df.copy()
df = pd.read_csv('/data/workspace_files/gdp.csv', decimal=",", delimiter=';')
gdp_df = df.copy()
df = pd.read_csv('/data/workspace_files/pop.csv', decimal=",", delimiter=';')
pop_df = df.copy()
df = pd.read_csv('/data/workspace_files/us-trade.csv', decimal=",", delimiter=';')
df['Exports'] = df['Exports']*1_000_000
df['Imports customs'] = df['Imports customs']*1_000_000
gdp_df = countries_df.merge(gdp_df, on=['Country Code'])
pop_df = countries_df.merge(pop_df, on=['Country Code'])
trade_df = pop_df.merge(df, on=['Country'])

countries_geocoder = geocode_countries(df['Country']) \
    .allow_ambiguous()

# With the data ready, let's first render the regular trade data to a map.
# 
# There are some todo items at this stage:
# 
# + Double check the countries map correctly, Australia seems awfully small though value seemed large.


ggplot(df) + geom_livemap() + \
        geom_polygon(aes(fill='Exports'), map=countries_geocoder, map_join=['Country']) + \
        ggtitle('Exports from USA') + \
        scale_fill_gradient(low="white", high="blue") + ggsize(760, 300)

# In the above map, we can see that the more blue the country, the more exports from the United States flow to the country.


ggplot(df) + geom_livemap() + \
        geom_polygon(aes(fill='Imports customs'), map=countries_geocoder, map_join=['Country']) + \
        ggtitle('Imports to USA, customs basis') + \
        scale_fill_gradient(low="white", high="red") + ggsize(760, 300)

# In the above map, we can see that the more red the country, the more imports the country ships to the United States.


df['Balance'] = df['Exports'] - df['Imports customs']
df['NormalizedBalance'] = df['Balance'] / df['Balance'].abs().max()
ggplot(df) + geom_livemap() + \
        geom_polygon(aes(fill='Balance'), map=countries_geocoder, map_join=['Country']) + \
        ggtitle('Trade Balance with USA') + \
        scale_fill_gradient2(low="red", high="blue") + ggsize(760, 300)

# The above map illustrates the trade balance between countries. The more blue the country, the "better" (surplus) the trade partner is to the United States. The more red the country, the "worse" (deficit) the trade partner is.


trade_df['AdjustedBalance'] = trade_df['Exports'] / trade_df['2019'] - trade_df['Imports customs'] / 328239523
trade_df['NormalizedAdjustedBalance'] = trade_df['AdjustedBalance']  / trade_df['AdjustedBalance'].abs().max()
ggplot(trade_df) + geom_livemap() + \
        geom_polygon(aes(fill='AdjustedBalance'), map=countries_geocoder, map_join=['Country']) + \
        ggtitle('Adjusted Trade Balance with USA') + \
        scale_fill_gradient2(low='red', high="blue") + ggsize(760, 300)
#trade_df['2019']

# The above map adjusts the trade balances to the populations. That is, the trade data is viewed as per capita. When viewing the difference below, you can see many countries that could initially be perceived as bad trading partners actually are really good partners.


