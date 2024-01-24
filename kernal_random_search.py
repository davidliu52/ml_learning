import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from scipy.stats import loguniform
import model_utils as mu
import build_4d_model_config as config
import seaborn as sns
## test
def apply_scaler(X_tr, Y_tr, X_te, Y_te):
    # Scalers for X and Y, fit on training data
    X_sc = StandardScaler().fit(X_tr)
    Y_sc = StandardScaler().fit(Y_tr.reshape(-1, 1))

    # Transform training data
    X_tr_scaled = X_sc.transform(X_tr)
    Y_tr_scaled = Y_sc.transform(Y_tr.reshape(-1, 1))

    # Transform test data
    X_te_scaled = X_sc.transform(X_te)
    Y_te_scaled = Y_sc.transform(Y_te.reshape(-1, 1))
    return X_tr_scaled, Y_tr_scaled, X_te_scaled, Y_te_scaled, X_sc, Y_sc

##testing-------------
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

def test_model_matrix_random_search(X, Y, random_state=None, n_iter=100):
    """
    Perform random search on degree and alpha of Kernel Ridge.

    Args:
        X (np.ndarray): Training features.
        Y (np.ndarray): Target features.
        random_state (int, optional): Random seed. Defaults to None.
        n_iter (int, optional): Number of parameter settings sampled. Defaults to 100.

    Returns:
        pd.DataFrame: Random search results.
    """

    # K-Folds cross-validator
    k = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=random_state)

    # Train test split
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X, Y, test_size=config.TEST_SIZE, shuffle=True, random_state=random_state
    )

    # scale features
    x, y, x_test, y_test, X_sc, Y_sc = apply_scaler(X_tr, Y_tr, X_te, Y_te)

    # Define the model
    model = KernelRidge(kernel="poly")

    # Define the parameter distributions
    param_distributions = {
        'degree': range(1, 6),  # Assuming you want to test degrees 1 to 5
        'alpha': np.random.uniform(low=0.05, high=0.95, size=100)  # Continuous distribution from 0.05 to 1
    }

    # RandomizedSearchCV
    random_search = RandomizedSearchCV(
        model, 
        param_distributions, 
        n_iter=n_iter, 
        cv=k, 
        random_state=random_state, 
        n_jobs=-1,
        refit=True,
        scoring='r2'

    )

    # Fit RandomizedSearchCV
    random_search.fit(x, y.ravel())

    # Getting the best model
    best_model = random_search.best_estimator_

    # Test best model on the test set
    test_score = best_model.score(x_test, y_test.ravel())
    best_params = random_search.best_params_
    results = pd.DataFrame()

    #print(best_params)
    # Save the best model
    if test_score > 0.5:
        name = f"{test_score:.4f}_d{best_model.degree}-a{best_model.alpha:.2f}-rs{random_state}--{best_params}"
        #print(name)
        results=[random_state,best_params['alpha'],best_params['degree'],test_score]
        print(results)
        # print(f"Saving model to {mu.model_path}...")
        # pickle.dump(best_model, open(mu.model_path.joinpath(f"{name}.p"), "wb"))

    return results


if __name__ == "__main__":
    # Set a random seed for reproducibility
    np.random.seed(42)

# Generate example data
    def generate_example_data(size=100):
        X = np.linspace(0, 10, size)  # Example feature (e.g., time)
        Y = 2 * X + 1 + np.random.normal(0, 1, size)  # Example target (linear relationship with noise)
        return X.reshape(-1, 1), Y

# Example data
    X, Y = generate_example_data()

    # data = mu.load_data()
    # X, Y, data = mu.prepare_data(data)
    if config.VISUALIZE:
        mu.visualize_data(X, Y)

    elif config.GRID_SEARCH:
        # degrees for kernel ridge grid search

        # loop over multiple random states
        states = np.random.randint(1000, size=50)
        results = []

        for random_state in states:
            # for each random state perform grid search on a kernel ridge model
            # and return the result
            result = test_model_matrix_random_search(X, Y, random_state=random_state)
            results.append(result)
        results=pd.DataFrame(results)
#        print("Random states used:", states)

        best_para = results[results.iloc[:,-1]== results.iloc[:,-1].max()]
        best_para = best_para.values.tolist()[0]

        print("best_para:", best_para)

        # Plotting results (adapt this as needed for your specific visualization requirements)
        fig, ax = plt.subplots(figsize=[8, 12])
        # for res in results:
        #     print(res.shape)
        #plt.bar(results.iloc[:,1], results.iloc[:,2], label=f"Random State: {results.iloc[:,0]}")
        ax=sns.stripplot(x=results.iloc[:,2], y= results.iloc[:,3],dodge=True, alpha=1, legend=True,
  )    
        #ax.set_ylim([0, 1])
        plt.xlabel("degree")
        plt.ylabel("score")
        parent=Path(r'D:\studydata\randomsearch')
        plt.savefig(parent.joinpath("performance.png"), dpi=150)

    else:
        print("No training method selected.")
