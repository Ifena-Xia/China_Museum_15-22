import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load GDP per capita data
gdp_df = pd.read_excel(".....GDP_per_capita.xlsx", index_col=0)  # Provinces as index, years as columns

# Load region-year data (museum statistics)
region_year_data = {}  # Dictionary to store yearly data
with pd.ExcelFile(".....region_year.xlsx") as xls:
    for sheet in xls.sheet_names:
        region_year_data[sheet] = pd.read_excel(xls, sheet_name=sheet, index_col=0)

# Combine yearly data into a single DataFrame
all_years_data = []
for year, df in region_year_data.items():
    df["Year"] = int(year)  # Add year column
    all_years_data.append(df)

museum_df = pd.concat(all_years_data)  # Merge all years into one DataFrame
museum_df.reset_index(inplace=True)  # Reset index to move province into a column
museum_df.rename(columns={"index": "Province"}, inplace=True)  # Rename province column

# Compute subsidy per square meter
museum_df["Subsidy_per_sqm"] = museum_df["Subsidy (10^6 RMB)"] / museum_df["Museum size (10^6 ㎡)"]

# Merge GDP data
gdp_long = gdp_df.stack().reset_index()
gdp_long.columns = ["Province", "Year", "GDP_per_capita"]
merged_df = pd.merge(museum_df, gdp_long, on=["Province", "Year"])

# check the result of merging
print(merged_df.columns)
# result: Index(['Province', 'Unnamed: 1', 'Count', 'Museum size (10^6 ㎡)',
      # 'Subsidy (10^6 RMB)', 'Expenditure (10^6 RMB)', 'Year',
      # 'Subsidy_per_sqm', 'GDP_per_capita'],
      # dtype='object')

# from the first description we notice the GDP_per_capita is treated like a string, convert it
print(merged_df["GDP_per_capita"].dtype)  
merged_df["GDP_per_capita"] = pd.to_numeric(merged_df["GDP_per_capita"], errors="coerce")
print(merged_df[["Subsidy_per_sqm", "GDP_per_capita", "Expenditure (10^6 RMB)"]].describe())
# Result: Index(['Province', 'Count', 'Museum size (10^6 ㎡)', 'Subsidy (10^6 RMB)',
      # 'Expenditure (10^6 RMB)', 'Year', 'Subsidy_per_sqm', 'GDP_per_capita'],
      #dtype='object')
'''
merged_df["A"]     # one column, returns a Series
merged_df[["A"]]   # one column, still a DataFrame
merged_df[["A", "B", "C"]]  # multiple columns as a DataFrame
'''

# Create region mapping
region_mapping = {
    "Beijing": "East", "Tianjin": "East", "Liaoning": "East", "Shanghai": "East", "Jiangsu": "East",
    "Zhejiang": "East", "Fujian": "East", "Shandong": "East", "Guangdong": "East",
    "Hebei": "Central", "Shanxi": "Central", "Heilongjiang": "Central", "Jilin": "Central",
    "Anhui": "Central", "Jiangxi": "Central", "Henan": "Central", "Hubei": "Central", "Hunan": "Central", "Hainan": "Central",
    "Inner Mongolia": "West", "Guangxi": "West", "Chongqing": "West", "Sichuan": "West", "Guizhou": "West", "Yunnan": "West",
    "Tibet": "West", "Shannxi": "West", "Gansu": "West", "Qinghai": "West", "Ningxia": "West", "Xinjiang": "West"
}

# Apply region mapping based on province name
merged_df["region"] = merged_df["Province"].map(region_mapping)

# The STAGE ONE distribution shows skewed
# log transform the data
from scipy.stats.mstats import winsorize
merged_df["log_Subsidy_per_sqm"] = np.log1p(merged_df["Subsidy_per_sqm"])
merged_df["log_Expenditure"] = np.log1p(merged_df["Expenditure (10^6 RMB)"])
merged_df["log_GDP_per_capita"] = np.log1p(merged_df["GDP_per_capita"])

'''
# STAGE ONE: overall describe the data by heatmap

provinces = merged_df["Province"].dropna().unique()
years = np.sort(merged_df["Year"].unique())
full_index = pd.MultiIndex.from_product([provinces, years], names=["Province", "Year"])

# Pivot tables for each variable
heatmap_logGDP = merged_df.pivot_table(
    index="Province", columns="Year", 
    values="log_GDP_per_capita", aggfunc="mean"
)
heatmap_logSubsidy = merged_df.pivot_table(
    index="Province", columns="Year", 
    values="log_Subsidy_per_sqm", aggfunc="mean"
)
heatmap_logExpenditure = merged_df.pivot_table(
    index="Province", columns="Year", 
    values="log_Expenditure", aggfunc="mean"
)

# Calculate the average log GDP for each province across years
avg_log_GDP = heatmap_logGDP.mean(axis=1)
# Sort provinces in descending order based on average log GDP
sorted_provinces_logGDP = avg_log_GDP.sort_values(ascending=False).index
# Reindex the pivot table accordingly
heatmap_logGDP_sorted = heatmap_logGDP.reindex(sorted_provinces_logGDP)

# For the Log Subsidy per sqm heatmap:
# Calculate the average log subsidy for each province (across all years)
avg_log_subsidy = heatmap_logSubsidy.mean(axis=1)
# Sort the provinces by average log subsidy in descending order
sorted_provinces_logSubsidy = avg_log_subsidy.sort_values(ascending=False).index
# Reindex the pivot table accordingly
heatmap_logSubsidy_sorted = heatmap_logSubsidy.reindex(sorted_provinces_logSubsidy)

# For the Log Expenditure heatmap:
# Calculate the average log expenditure for each province (across all years)
avg_log_expenditure = heatmap_logExpenditure.mean(axis=1)
# Sort the provinces by average log expenditure in descending order
sorted_provinces_logExpenditure = avg_log_expenditure.sort_values(ascending=False).index
# Reindex the pivot table accordingly
heatmap_logExpenditure_sorted = heatmap_logExpenditure.reindex(sorted_provinces_logExpenditure)

# Sort provinces based on 2016 values for each metric
sorted_provinces_GDP = heatmap_logGDP[2016].sort_values(ascending=False).index
sorted_provinces_Subsidy = heatmap_logSubsidy[2016].sort_values(ascending=False).index
sorted_provinces_Expenditure = heatmap_logExpenditure[2016].sort_values(ascending=False).index

# Reindex the pivot tables using the sorted order
heatmap_logGDP_sorted = heatmap_logGDP.reindex(sorted_provinces_GDP)
heatmap_logSubsidy_sorted = heatmap_logSubsidy.reindex(sorted_provinces_Subsidy)
heatmap_logExpenditure_sorted = heatmap_logExpenditure.reindex(sorted_provinces_Expenditure)
                                                               
# Plot for Log GDP per capita sorted by average log GDP
plt.figure(figsize=(12, 8))
sns.heatmap(
    heatmap_logGDP_sorted, 
    cmap="viridis", 
    linewidths=0.3, linecolor='gray',
    cbar_kws={'label': 'Log GDP per Capita'}
)
plt.title("Log GDP per Capita by Province and Year\n(Sorted by Average Log GDP per Capita)")
plt.xlabel("Year")
plt.ylabel("Province")
plt.tight_layout()
plt.show()

# Plot for Log Subsidy per sqm sorted by average log subsidy
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_logSubsidy_sorted, cmap="cividis", linewidths=0.3, linecolor='gray',
            cbar_kws={'label': 'Log Subsidy per sqm'})
plt.title("Log Subsidy per sqm by Province and Year\n(Sorted by Average Log Subsidy)")
plt.xlabel("Year")
plt.ylabel("Province")
plt.tight_layout()
plt.show()

# Plot for Log Expenditure sorted by average log expenditure
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_logExpenditure_sorted, cmap="YlGnBu", linewidths=0.3, linecolor='gray',
            cbar_kws={'label': 'Log Museum Expenditure'})
plt.title("Log Museum Expenditure by Province and Year\n(Sorted by Average Log Expenditure)")
plt.xlabel("Year")
plt.ylabel("Province")
plt.tight_layout()
plt.show()
'''

'''
# STAGE TWO: overall describe the data by bar chart and box plot

# bar chart over time

# Aggregate mean values by Year and region
region_year_avg = merged_df.groupby(["Year", "region"]).agg({
    "GDP_per_capita": "mean",
    "Subsidy_per_sqm": "mean",
    "Expenditure (10^6 RMB)": "mean"
}).reset_index()

region_palette = {
    "East": "#FDBF6F",    # Soft Orange
    "Central": "#A6CEE3", # Soft Blue
    "West": "#B2DF8A"     # Soft Green
}

plt.figure(figsize=(10, 6))
sns.barplot(
    data=region_year_avg, 
    x="Year", 
    y="GDP_per_capita", 
    hue="region", 
    palette=region_palette
)
plt.title("Average GDP per Capita by Region and Year")
plt.ylabel("GDP per Capita (yuan)")
plt.xlabel("Year")
plt.legend(title="Region")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(
    data=region_year_avg, 
    x="Year", 
    y="Subsidy_per_sqm", 
    hue="region", 
    palette=region_palette
)
plt.title("Average Subsidy per sqm by Region and Year")
plt.ylabel("Subsidy per sqm (yuan)")
plt.xlabel("Year")
plt.legend(title="Region")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(
    data=region_year_avg, 
    x="Year", 
    y="Expenditure (10^6 RMB)", 
    hue="region", 
    palette=region_palette
)
plt.title("Average Museum Expenditure by Region and Year")
plt.ylabel("Expenditure (10^6 RMB)")
plt.xlabel("Year")
plt.legend(title="Region")
plt.tight_layout()
plt.show()

# box plots
plt.figure(figsize=(12, 6))

# Box plots for Subsidy per square meter, GDP per capita, and Museum expenditure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.boxplot(data=merged_df, x="log_Subsidy_per_sqm", ax=axes[0])
axes[0].set_title("Distribution of Log Subsidy per Square Meter")

sns.boxplot(data=merged_df, x="log_GDP_per_capita", ax=axes[1])
axes[1].set_title("Distribution of Log GDP per Capita")

sns.boxplot(data=merged_df, x="log_Expenditure", ax=axes[2])
axes[2].set_title("Distribution of Log Museum Expenditure")

plt.tight_layout()
plt.show()
'''

'''
# STAGE Three: overall lmplot

# Log GDP vs. Log Subsidy
sns.lmplot(
    data=merged_df,
    x="log_GDP_per_capita",
    y="log_Subsidy_per_sqm",
    col="Year",         # one subplot per year
    hue="region",       # different colors for each region
    markers="o",
    col_wrap=4,   # Wrap the plots into rows of 4 columns      
    height=3,
    scatter_kws={'alpha': 0.6}, # 60% visible, 40% transparent
    ci=None             # set to None if you prefer not to show confidence intervals
)

# Log GDP vs. Log Expenditure
plt.subplots_adjust(top=0.9)
plt.suptitle("Log GDP per Capita vs. Log Subsidy per sqm by Year and Region")
plt.show()

sns.lmplot(
    data=merged_df,
    x="log_Subsidy_per_sqm",
    y="log_Expenditure",
    col="Year",
    hue="region",
    markers="o",
    col_wrap=4,
    height=3,
    scatter_kws={'alpha': 0.6},
    ci=None
)

plt.subplots_adjust(top=0.9)
plt.suptitle("Log Subsidy per sqm vs. Log Expenditure by Year and Region")
plt.show()

# Log GDP vs. Log Expenditure
sns.lmplot(
    data=merged_df,
    x="log_GDP_per_capita",
    y="log_Expenditure",
    col="Year",
    hue="region",
    markers="o",
    col_wrap=4,
    height=3,
    scatter_kws={'alpha': 0.6},
    ci=None
)

plt.subplots_adjust(top=0.9)
plt.suptitle("Log GDP per Capita vs. Log Expenditure by Year and Region")
plt.show()

# result: no clear, stable linear relationship
'''

'''
# STAGE FOUR: check the correlation

for y in merged_df["Year"].unique():
    subset = merged_df[merged_df["Year"] == y]
    corr = subset["log_GDP_per_capita"].corr(subset["log_Subsidy_per_sqm"])
    print(f"Year {y}, Corr(GDP, Subsidy): {corr:.3f}")

for y in merged_df["Year"].unique():
    subset = merged_df[merged_df["Year"] == y]
    corr = subset["log_Subsidy_per_sqm"].corr(subset["log_Expenditure"])
    print(f"Year {y}, Corr(Subsidy, Expenditure): {corr:.3f}")

for y in merged_df["Year"].unique():
    subset = merged_df[merged_df["Year"] == y]
    corr = subset["log_GDP_per_capita"].corr(subset["log_Expenditure"])
    print(f"Year {y}, Corr(GDP, Expenditure): {corr:.3f}")

# corr result of Pearson correlation coefficient for each year
# Year 2016, Corr(GDP, Subsidy): 0.327
# Year 2017, Corr(GDP, Subsidy): 0.092
# Year 2018, Corr(GDP, Subsidy): -0.011
# Year 2019, Corr(GDP, Subsidy): 0.133
# Year 2020, Corr(GDP, Subsidy): 0.158
# Year 2021, Corr(GDP, Subsidy): 0.148
# Year 2022, Corr(GDP, Subsidy): 0.196
# Year 2016, Corr(Subsidy, Expenditure): 0.037
# Year 2017, Corr(Subsidy, Expenditure): 0.023
# Year 2018, Corr(Subsidy, Expenditure): -0.084
# Year 2019, Corr(Subsidy, Expenditure): 0.031
# Year 2020, Corr(Subsidy, Expenditure): 0.064
# Year 2021, Corr(Subsidy, Expenditure): 0.326
# Year 2022, Corr(GDP, Expenditure): -0.104
# Year 2016, Corr(GDP, Expenditure): 0.402
# Year 2017, Corr(GDP, Expenditure): 0.464
# Year 2018, Corr(GDP, Expenditure): 0.503
# Year 2019, Corr(GDP, Expenditure): 0.560
# Year 2020, Corr(GDP, Expenditure): 0.564
# Year 2021, Corr(GDP, Expenditure): 0.452
# Year 2022, Corr(GDP, Expenditure): 0.485
# incosistent and instable, GDP and Expenditure not robust enough

# linear relationship CONTROLLING YEARS
import statsmodels.formula.api as smf

# Model: log_Subsidy_per_sqm as a function of log_GDP_per_capita, controlling for year effects.
model = smf.ols("log_Subsidy_per_sqm ~ log_GDP_per_capita + C(Year)", data=merged_df).fit()
print(model.summary())
# The model looks like log_Subsidy_per_sqmi = β0 + β1 * log_GDP_per_capitai + γt * Yeart + εi
# Take year as a parameter

# Model: log_Expenditure as a function of log_Subsidy_per_sqm, controlling for year effects.
model = smf.ols("log_Expenditure ~ log_Subsidy_per_sqm + C(Year)", data=merged_df).fit()
print(model.summary())

# Model: log_Expenditure as a function of log_GDP_per_capita, controlling for year effects.
model = smf.ols("log_Expenditure ~ log_GDP_per_capita + C(Year)", data=merged_df).fit()
print(model.summary())

# Result: nonrobust for all
'''

'''
# segemented by regions
# List of unique regions in your data
import statsmodels.formula.api as smf
from scipy.stats import pearsonr

regions = merged_df["region"].unique()

for reg in regions:
    subset = merged_df[merged_df["region"] == reg]
# This isolates data by region so each regression is region-specific

    # Check if the subset is empty or if any variable has no valid values
    if subset.empty:
        print(f"No data for region {reg}. Skipping.")
        continue
    if (subset["log_GDP_per_capita"].dropna().empty or 
        subset["log_Subsidy_per_sqm"].dropna().empty or 
        subset["log_Expenditure"].dropna().empty):
        print(f"Not enough data for region {reg}. Skipping.")
        continue
    
    print("="*60)
    print(f"Region: {reg}")
    
    # Relationship 1: log_GDP_per_capita vs. log_Subsidy_per_sqm
    corr1 = subset["log_GDP_per_capita"].corr(subset["log_Subsidy_per_sqm"])
    print(f"Pearson Corr (log_GDP vs. log_Subsidy): {corr1:.3f}")
    
    model1 = smf.ols("log_Subsidy_per_sqm ~ log_GDP_per_capita", data=subset).fit()
    print("\nRegression: log_Subsidy_per_sqm ~ log_GDP_per_capita")
    print(model1.summary())
    
    # Relationship 2: log_Subsidy_per_sqm vs. log_Expenditure
    corr2 = subset["log_Subsidy_per_sqm"].corr(subset["log_Expenditure"])
    print(f"\nPearson Corr (log_Subsidy vs. log_Expenditure): {corr2:.3f}")
    
    model2 = smf.ols("log_Expenditure ~ log_Subsidy_per_sqm", data=subset).fit()
    print("\nRegression: log_Expenditure ~ log_Subsidy_per_sqm")
    print(model2.summary())
    
    # Relationship 3: log_GDP_per_capita vs. log_Expenditure
    corr3 = subset["log_GDP_per_capita"].corr(subset["log_Expenditure"])
    print(f"\nPearson Corr (log_GDP vs. log_Expenditure): {corr3:.3f}")
    
    model3 = smf.ols("log_Expenditure ~ log_GDP_per_capita", data=subset).fit()
    print("\nRegression: log_Expenditure ~ log_GDP_per_capita")
    print(model3.summary())
    print("="*60, "\n")

# result: none of the regions show robust corr
'''

'''
# STAGE FIVE: SIZE
merged_df["log_Museum_size"] = np.log1p(merged_df["Museum size (10^6 ㎡)"])
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=merged_df,
    x="log_GDP_per_capita",
    y="log_Museum_size",
    hue="region",
    alpha=0.6
)
sns.regplot(
    data=merged_df,
    x="log_GDP_per_capita",
    y="log_Museum_size",
    scatter=False,
    color="black",
    line_kws={"linestyle": "--"}
)
plt.title("Log GDP per Capita vs. Log Museum Size")
plt.xlabel("Log GDP per Capita")
plt.ylabel("plt.ylabel('Museum size (10^6 square meters')")
plt.grid(True)
plt.tight_layout()
plt.show()

from scipy.stats import pearsonr, spearmanr
import statsmodels.formula.api as smf
plt.rcParams['font.family'] = 'Arial Unicode MS'

# test corr
# Pearson (linear)
pearson_corr, pearson_p = pearsonr(merged_df["log_GDP_per_capita"], merged_df["log_Museum_size"])
print(f"Pearson correlation: {pearson_corr:.3f}, p = {pearson_p:.3f}")
# Spearman (monotonic)
spearman_corr, spearman_p = spearmanr(merged_df["log_GDP_per_capita"], merged_df["log_Museum_size"])
print(f"Spearman correlation: {spearman_corr:.3f}, p = {spearman_p:.3f}")
# result:
# Pearson correlation: 0.396, p = 0.000
# Spearman correlation: 0.465, p = 0.000

# control for year
model = smf.ols("log_Museum_size ~ log_GDP_per_capita + C(Year)", data=merged_df).fit()
print(model.summary())
# Controlling for national trends, do provinces with higher GDP per capita than others in the same year have larger museums?
#result: nonrobust

# add province fixed effects
model = smf.ols("log_Museum_size ~ log_GDP_per_capita + C(Year) + C(Province)", data=merged_df).fit()
# When GDP goes up within a province, do museum sizes grow?
print(model.summary())
# results: non robust

# try regression without control
model = smf.ols("log_Museum_size ~ log_GDP_per_capita", data=merged_df).fit()
print(model.summary())
# result: non robust
# Wealthier provinces tend to have bigger museums (correlation),
# but becoming wealthier doesn't make them build bigger museums (regression).
'''


'''
========= TRY SOME IMAGE ===========
# try the hexbin plot
plt.hexbin(merged_df["log_Subsidy_per_sqm"], merged_df["GDP_per_capita"], gridsize=50, cmap="Blues", mincnt=1)
plt.colorbar(label="Density")
plt.xlabel("Log(1 + Subsidy per sqm)")
plt.ylabel("GDP per Capita")
plt.title("Hexbin Plot of Log-Transformed Subsidy and GDP")
plt.show()

# try scatterplot by region
sns.scatterplot(
    data=merged_df,
    x="log_Subsidy_per_sqm",
    y="GDP_per_capita",
    hue="region",
    palette="Set2",
    alpha=0.7
)
plt.title("Log(1 + Subsidy per sqm) vs. GDP per Capita by Region")
plt.xlabel("Log(1 + Subsidy per sqm)")
plt.ylabel("GDP per Capita")
plt.grid(True)
plt.tight_layout()
plt.show()
'''

