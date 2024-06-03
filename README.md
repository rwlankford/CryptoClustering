# CryptoClustering
Module 11 Challenge

# Cryptocurrency Clustering Analysis

This project aims to cluster various cryptocurrencies based on their price changes over different time periods. The analysis involves standardizing the data, determining the optimal number of clusters using the K-means algorithm, and visualizing the results. Additionally, Principal Component Analysis (PCA) is used to reduce the data dimensions and further optimize the clustering process.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Data Preparation](#data-preparation)
4. [Finding the Best Value for k](#finding-the-best-value-for-k)
5. [Clustering Cryptocurrencies](#clustering-cryptocurrencies)
6. [Optimizing Clusters with PCA](#optimizing-clusters-with-pca)
7. [Determining Feature Weights](#determining-feature-weights)
8. [Contributing](#contributing)
9. [License](#license)

## Installation

To run this project, you need to have Python installed along with the following libraries:

- pandas
- hvplot
- scikit-learn

You can install these libraries using pip:

```bash
pip install pandas hvplot scikit-learn
```

## Usage

1. Clone the repository.
2. Navigate to the project directory.
3. Run the Jupyter Notebook or Python script.

## Data Preparation

1. Load the cryptocurrency market data into a Pandas DataFrame and set the index to `coin_id`.

    ```python
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    market_data_df = pd.read_csv("Resources/crypto_market_data.csv", index_col="coin_id")
    ```

2. Normalize the data using `StandardScaler`.

    ```python
    market_data_scaled = StandardScaler().fit_transform(market_data_df)
    df_market_data_scaled = pd.DataFrame(market_data_scaled, columns=market_data_df.columns)
    df_market_data_scaled["coin_id"] = market_data_df.index
    df_market_data_scaled = df_market_data_scaled.set_index("coin_id")
    ```

## Finding the Best Value for k

1. Create a range of k values to try (1 to 11).

    ```python
    k = range(1, 11)
    inertia_values = []
    ```

2. Compute the inertia for each k value using K-means.

    ```python
    from sklearn.cluster import KMeans

    for i in k:
        model = KMeans(n_clusters=i)
        model.fit(df_market_data_scaled)
        inertia_values.append(model.inertia_)
    ```

3. Plot the Elbow curve to identify the optimal k value.

    ```python
    import pandas as pd

    inertia_df = pd.DataFrame({"k": k, "inertia": inertia_values})
    inertia_df.plot(x="k", y="inertia", title="Elbow Curve", xticks=k)
    ```

Based on the Elbow curve, the best value for k is determined to be 4.

## Clustering Cryptocurrencies

1. Initialize and fit the K-means model using the optimal k value.

    ```python
    k = 4
    kmeans_model = KMeans(n_clusters=k)
    kmeans_model.fit(df_market_data_scaled)
    ```

2. Predict the clusters and add the cluster labels to the DataFrame.

    ```python
    df_market_data_scaled["predicted_clusters"] = kmeans_model.predict(df_market_data_scaled)
    ```

3. Visualize the clusters.

    ```python
    df_market_data_scaled.plot.scatter(
        x="price_change_percentage_24h",
        y="price_change_percentage_7d",
        c="predicted_clusters",
        colormap="rainbow"
    )
    ```

## Optimizing Clusters with PCA

1. Apply PCA to reduce the data to three principal components.

    ```python
    from sklearn.decomposition import PCA

    pca_model = PCA(n_components=3)
    market_data_pca = pca_model.fit_transform(df_market_data_scaled)
    df_market_data_pca = pd.DataFrame(data=market_data_pca, columns=["PC1", "PC2", "PC3"])
    df_market_data_pca["coin_id"] = df_market_data_scaled.index
    df_market_data_pca = df_market_data_pca.set_index("coin_id")
    ```

2. Determine the optimal k value using the PCA data and plot the Elbow curve.

    ```python
    inertia_pca_values = []

    for i in range(1, 12):
        model = KMeans(n_clusters=i)
        model.fit(df_market_data_pca)
        inertia_pca_values.append(model.inertia_)

    inertia_pca_df = pd.DataFrame({"k": range(1, 12), "inertia": inertia_pca_values})
    inertia_pca_df.plot.line(x="k", y="inertia", title="Elbow Curve: Inertia vs. Number of Clusters")
    ```

3. Fit the K-means model using the optimal k value on the PCA data and visualize the clusters.

    ```python
    km_model_pca = KMeans(n_clusters=4)
    km_model_pca.fit(df_market_data_pca)
    crypto_clusters_pca = km_model_pca.predict(df_market_data_pca)
    df_market_data_pca["predicted_clusters"] = crypto_clusters_pca

    df_market_data_pca.hvplot.scatter(
        x="PC1",
        y="PC2",
        by="predicted_clusters",
        hover_cols=["coin_id"],
        title="Cryptocurrency Clusters",
        colormap="rainbow",
        width=800,
        height=600
    )
    ```

## Determining Feature Weights

1. Determine the weights of each feature on each principal component.

    ```python
    pca_weights = pd.DataFrame(pca_model.components_.T, columns=['PC1', 'PC2', 'PC3'], index=df_market_data_scaled.columns)
    ```

The features with the strongest influence on each principal component are:

- **PC1**: Strongest Positive = 200d, Strongest Negative = 24h
- **PC2**: Strongest Positive = 30d, Strongest Negative = 1yr
- **PC3**: Strongest Positive = 7d, Strongest Negative = 60d

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
