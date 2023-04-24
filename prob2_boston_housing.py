# %%
import numpy as np
from bo_util import *
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# %%
# Define the objective function and  intial hyperparameters
def get_data():
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    boston = pd.read_csv('data/housing.csv', header=None, delimiter=r"\s+", names=column_names)

    # Create a pandas DataFrame from the dataset
    df = pd.DataFrame(boston)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df.drop('MEDV', axis=1), df['MEDV'], test_size=0.2, random_state=42)
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def build_model(hyperparams):
    n_layers, n_nodes, learning_rate, dropout_rate = hyperparams
    n_layers = int(n_layers)
    n_nodes = int(n_nodes)
    model = Sequential()
    model.add(Dense(n_nodes, activation='relu', input_dim=13))
    for i in range(n_layers-1):
        model.add(Dense(n_nodes, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mse')
    return model

def worker_obj(params):
    X_train_scaled, X_test_scaled, y_train, y_test = get_data()
    start_time = time.time()
    model = build_model(params)
    model.fit(X_train_scaled, y_train, epochs=20, validation_split=0.2, callbacks=[EarlyStopping(patience=5)], verbose=0)
    score = model.evaluate(X_test_scaled, y_test, verbose=0)
    end_time = time.time()
    elapsed_time = (end_time-start_time) # nanoseconds
    return score, elapsed_time

def obj(x):    
    elapsed_times = []
    vals = []
    params_total = x.reshape(-1, 4)

    for params in params_total:
        params = params.reshape(-1)
        val, elapsed_time = worker_obj(params)
        vals.append(-val)
        elapsed_times.append(elapsed_time)
   
    return vals, elapsed_times
# %% BO
if __name__ == "__main__":  # confirms that the code is under main function
    # %% BO
    # Iteration parameters
    n_loop = 10
    # BO parameters
    n_max_iter = 30
    n_init_samples=5
    n_cand=10
    n_worker=5
    n_mcmc=15
    pbounds = {'n_layers':(1, 5),
            'n_nodes': (32, 128) , 
            'learning_rate': (0.0001, 0.1), 
            'dropout_rate': (0, 0.4)}
    best_each_loop = []
    best_results = []

    consider_time = True
    kernel_fun = 'matern' # 'matern' / 'se'
    use_pending = True
    minus_obj_fun = True
    best_each_loop = run_bo(obj, n_loop, pbounds, n_init_samples, n_worker, n_cand, n_mcmc, n_max_iter, consider_time, kernel_fun, use_pending, minus_obj_fun)
    np.savetxt('result/prob2_bo.csv', best_each_loop, delimiter=",")

    #%% Random search
    best_each_loop = []
    for _ in range(n_loop):
        s_rand = Samples(obj, bounds=pbounds, n_init_samples=n_init_samples, n_worker=n_worker, n_cand=n_cand, n_mcmc=n_mcmc)
        s_rand.random_sampling(n_max_iter, init_sampling=True)
        y = -s_rand.y0
        best_single_loop = np.zeros_like(y)
        best_single_loop[0] = y[0] 
        for i in range(1, len(y)):
            best_single_loop[i] = min(best_single_loop[i-1], y[i]) # The current value in b is the minimum of the previous value in b and the current value in a
        best_each_loop.append(best_single_loop)

    np.savetxt('result/prob2_rnd.csv', best_each_loop, delimiter=",")