# ==========================================================
# Netflix Titles Dataset - Exploratory Data Analysis (EDA)
# ==========================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression

# Plot Settings
plt.style.use('ggplot')
sns.set(style="whitegrid")

# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv("C:\\Users\\ksche\\netflix_titles.csv")

print("="*60)
print("First 5 Rows")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

# ==========================================================
# Data Cleaning
# ==========================================================

# Remove leading/trailing spaces
df['date_added'] = df['date_added'].str.strip()

# Convert to datetime
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')

# Fill missing values
df['director'] = df['director'].fillna("Unknown")
df['cast'] = df['cast'].fillna("Unknown")
df['country'] = df['country'].fillna("Unknown")
df['rating'] = df['rating'].fillna(df['rating'].mode()[0])
df['duration'] = df['duration'].fillna("Unknown")

print("\nMissing Values After Cleaning")
print(df.isnull().sum())

# ==========================================================
# Feature Engineering
# ==========================================================

# Movie Flag
df['is_movie'] = df['type'].apply(lambda x: 1 if x == "Movie" else 0)

# Release Decade
df['release_decade'] = (df['release_year'] // 10) * 10

# Content Type
df['content_type'] = df['type']

# Year Added
df['year_added'] = df['date_added'].dt.year

print("\nNew Features")
print(df[['type','is_movie','release_decade','year_added']].head())

# ==========================================================
# Top 5 Countries
# ==========================================================

countries = df['country'].str.split(',').explode().str.strip()

top5_countries = countries.value_counts().head(5)

print("\nTop 5 Countries")
print(top5_countries)

# ==========================================================
# Top 10 Genres
# ==========================================================

genres = df['listed_in'].str.split(',').explode().str.strip()

top10_genres = genres.value_counts().head(10)

print("\nTop 10 Genres")
print(top10_genres)

# ==========================================================
# Visualization 1
# Movie vs TV Shows
# ==========================================================

plt.figure(figsize=(6,6))

df['type'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    colors=['skyblue','orange']
)

plt.title("Movie vs TV Shows")
plt.ylabel("")
plt.show()

# ==========================================================
# Visualization 2
# Content Released Per Year
# ==========================================================

release = df.groupby('release_year').size()

plt.figure(figsize=(12,5))

plt.plot(release.index, release.values, marker='o')

plt.title("Content Released Per Year")

plt.xlabel("Release Year")

plt.ylabel("Number of Titles")

plt.show()

# ==========================================================
# Visualization 3
# Top 10 Genres
# ==========================================================

plt.figure(figsize=(12,6))

sns.barplot(
    x=top10_genres.values,
    y=top10_genres.index,
    hue=top10_genres.index,
    palette='viridis',
    legend=False
)

plt.title("Top 10 Genres")

plt.xlabel("Count")

plt.ylabel("Genre")

plt.show()

# ==========================================================
# Visualization 4
# Top 5 Countries
# ==========================================================

plt.figure(figsize=(8,5))

sns.barplot(
    x=top5_countries.values,
    y=top5_countries.index,
    hue=top5_countries.index,
    palette='Set2',
    legend=False
)

plt.title("Top 5 Countries")

plt.xlabel("Number of Titles")

plt.ylabel("Country")

plt.show()

# ==========================================================
# Visualization 5
# Heatmap - Country vs Content Volume
# ==========================================================

# Create a copy
country_df = df.copy()

# Split multiple countries into lists
country_df['country'] = country_df['country'].str.split(',')

# Convert each country into a separate row
country_df = country_df.explode('country')

# Remove extra spaces
country_df['country'] = country_df['country'].str.strip()

# Reset index (important for newer pandas versions)
country_df = country_df.reset_index(drop=True)

# Create summary table using groupby
heat = (
    country_df
    .groupby(['country', 'type'])
    .size()
    .unstack(fill_value=0)
)

# Show top 10 countries based on total content
heat['Total'] = heat.sum(axis=1)
heat = heat.sort_values(by='Total', ascending=False).head(10)
heat = heat.drop(columns='Total')

# Plot Heatmap
plt.figure(figsize=(8,6))

sns.heatmap(
    heat,
    annot=True,
    cmap='YlGnBu',
    fmt='d',
    linewidths=0.5
)

plt.title("Top 10 Countries vs Content Type")
plt.xlabel("Content Type")
plt.ylabel("Country")

plt.show()

# ==========================================================
# Visualization 6
# Content by Release Decade
# ==========================================================

plt.figure(figsize=(10,5))

sns.countplot(
    x='release_decade',
    data=df,
    hue='release_decade',
    palette='coolwarm',
    legend=False
)

plt.title("Content by Release Decade")

plt.xlabel("Release Decade")

plt.ylabel("Count")

plt.show()

# ==========================================================
# Summary Statistics
# ==========================================================

print("\nMovie vs TV Show")

print(df['type'].value_counts())

print("\nRelease Decades")

print(df['release_decade'].value_counts().sort_index())

print("\nTop Countries")

print(top5_countries)

print("\nTop Genres")

print(top10_genres)

# ==========================================================
# Bonus
# Trend Prediction
# ==========================================================

trend = df.groupby('release_year').size().reset_index(name='count')

X = trend[['release_year']]

y = trend['count']

model = LinearRegression()

model.fit(X, y)

future = pd.DataFrame({
    'release_year': [
        trend['release_year'].max()+1,
        trend['release_year'].max()+2
    ]
})

future['Predicted_Content'] = model.predict(future).astype(int)

print("\nPredicted Content For Next Two Years")

print(future)

# Plot Prediction

plt.figure(figsize=(12,5))

plt.plot(
    trend['release_year'],
    trend['count'],
    marker='o',
    label='Actual'
)

plt.plot(
    future['release_year'],
    future['Predicted_Content'],
    marker='o',
    linestyle='--',
    color='red',
    label='Prediction'
)

plt.title("Netflix Content Trend Prediction")

plt.xlabel("Release Year")

plt.ylabel("Number of Titles")

plt.legend()

plt.show()

# ==========================================================
# Final Insights
# ==========================================================

print("\n" + "="*60)
print("FINAL INSIGHTS")
print("="*60)

print("1. Movies dominate Netflix's library compared to TV Shows.")
print("2. Content production increased significantly after 2015.")
print("3. International Movies, Dramas, and Comedies are the most popular genres.")
print("4. The United States contributes the highest number of titles.")
print("5. Netflix has expanded its content library rapidly over the years.")
print("6. The simple linear regression model predicts continued growth in content releases over the next two years.")
print("="*60)