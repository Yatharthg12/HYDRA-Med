from src.data_processing import load_and_clean_data, prepare_features
from src.baseline_ml import run_logistic_regression
from src.pca_knn import run_pca_knn


def main():
    df = load_and_clean_data(filepath="data/diabetic_data_reduced.csv")
    X, y = prepare_features(df)

    print("\n--- Logistic Regression ---")
    run_logistic_regression(X, y)

    print("\n--- PCA + kNN ---")
    run_pca_knn(X, y)


if __name__ == "__main__":
    main()
