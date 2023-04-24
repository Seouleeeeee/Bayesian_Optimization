# %%
import numpy as np
from bo_util import *
from scipy.optimize import minimize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import fashion_mnist


# %%
# Define the objective function and  intial hyperparameters
def build_model(hyperparams):
    n_filters_1, n_filters_2, n_filters_3, activation, n_units, dropout_rate = hyperparams
    if activation==0:
        activation = 'relu'
    elif activation==1:
        activation = 'tanh'
    else:
        activation = 'sigmoid'
    model = Sequential([
        Conv2D(n_filters_1, (3,3), activation=activation, input_shape=(28,28,1)),
        MaxPooling2D((2,2)),
        Conv2D(n_filters_2, (3,3), activation=activation),
        MaxPooling2D((2,2)),
        Conv2D(n_filters_3, (3,3), activation=activation),
        Flatten(),
        Dense(n_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(10, activation='softmax')
    ])
    return model

def worker_obj(params):
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)


    start_time = time.time()
    params = params.reshape(-1)
    model = build_model(params)
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, validation_split=0.3, verbose=0)
    score = model.evaluate(X_test, y_test, verbose=0)[1]
    end_time = time.time()
    elapsed_time = (end_time-start_time) # nanoseconds
    return score, elapsed_time

def obj(x):    
    x = x.reshape(-1, 6)
    elapsed_times = []
    vals = []

    for _, (params) in enumerate(x):
        val, elapsed_time = worker_obj(params)
        vals.append(val)
        elapsed_times.append(elapsed_time)

    return vals, elapsed_times
# %% BO
if __name__ == "__main__":  # confirms that the code is under main function
    # %% BO
    # Iteration parameters
    n_loop = 7
    # BO parameters
    n_max_iter = 15
    n_init_samples=5
    n_cand=10
    n_worker=5
    n_mcmc=15
    best_each_loop = []
    best_results = []

    consider_time = True
    kernel_fun = 'matern' # 'matern' / 'se'
    use_pending = True
    minus_obj_fun = False
    best_each_loop, period_each_loop = run_bo(obj, n_loop, None, n_init_samples, n_worker, n_cand, n_mcmc, n_max_iter, consider_time, kernel_fun, use_pending, minus_obj_fun, prob3=True)
    np.savetxt('result/prob3_bo_ver1.csv', best_each_loop, delimiter=",")
    np.savetxt('result/prob3_bo_time_ver1.csv', period_each_loop, delimiter=",")

    consider_time = False
    kernel_fun = 'matern' # 'matern' / 'se'
    use_pending = True
    minus_obj_fun = False
    best_each_loop, period_each_loop = run_bo(obj, n_loop, None, n_init_samples, n_worker, n_cand, n_mcmc, n_max_iter, consider_time, kernel_fun, use_pending, minus_obj_fun, prob3=True)
    np.savetxt('result/prob3_bo_ver2.csv', best_each_loop, delimiter=",")
    np.savetxt('result/prob3_bo_time_ver2.csv', period_each_loop, delimiter=",")

    consider_time = True
    kernel_fun = 'se' # 'matern' / 'se'
    use_pending = True
    minus_obj_fun = False
    best_each_loop, period_each_loop = run_bo(obj, n_loop, None, n_init_samples, n_worker, n_cand, n_mcmc, n_max_iter, consider_time, kernel_fun, use_pending, minus_obj_fun, prob3=True)
    np.savetxt('result/prob3_bo_ver3.csv', best_each_loop, delimiter=",")
    np.savetxt('result/prob3_bo_time_ver3.csv', period_each_loop, delimiter=",")

    consider_time = True
    kernel_fun = 'matern' # 'matern' / 'se'
    use_pending = False
    minus_obj_fun = False
    best_each_loop, period_each_loop = run_bo(obj, n_loop, None, n_init_samples, n_worker, n_cand, n_mcmc, n_max_iter, consider_time, kernel_fun, use_pending, minus_obj_fun, prob3=True)
    np.savetxt('result/prob3_bo_ver4.csv', best_each_loop, delimiter=",")
    np.savetxt('result/prob3_bo_time_ver4.csv', period_each_loop, delimiter=",")

    #%% Random search
    best_each_loop = []
    period_each_loop = []
    for _ in range(n_loop):
        params = np.vstack((np.random.choice(np.arange(16, 128, 16), (1, n_max_iter)),
                np.random.choice(np.arange(16, 128, 16), (1, n_max_iter)),
                np.random.choice(np.arange(16, 128, 16), (1, n_max_iter)),
                np.random.choice([0, 1, 2], (1, n_max_iter)),
                np.random.choice(np.arange(16, 256, 16), (1, n_max_iter)),
                np.random.choice(np.arange(0.1, 0.6, 0.1), (1, n_max_iter))
                )).T
        best_single_loop, period_single_loop = obj(params)
        best_each_loop.append(best_single_loop)
        period_each_loop.append(period_single_loop)

    np.savetxt('result/prob3_rnd.csv', best_each_loop, delimiter=",")
    np.savetxt('result/prob3_rnd_time.csv', period_each_loop, delimiter=",")