# %%
import numpy as np
from bo_util import *
from scipy.optimize import minimize

# %%
# Define the objective function and  intial hyperparameters
def branin_hoo(x):
    x1, x2 = x
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    f = a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s
    return f

def branin_hoo_2(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    f = a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s
    return f

def obj(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
    x1=x[:, 0]
    x2=x[:, 1]
    elapsed_times = np.zeros((x.shape[0]))
    vals = np.zeros((x.shape[0]))
    for i, (x1, x2) in enumerate(x):
        start_time = time.perf_counter()
        a=1
        b=5.1/(4*(np.pi**2))
        c=5/(np.pi)
        r=6
        s=10
        t=1/(8*np.pi)
        end_time = time.perf_counter()
        elapsed_times[i] = (end_time-start_time)*(1e+6) # nanoseconds
        val=a*(x2-b*(x1**2)+c*x1-r)**2 + s*(1-t)*np.cos(x1)+s
        vals[i] = -val # Make minimization prob to maximization prob

    return vals, elapsed_times 

def callback(x):
    best_single_loop.append(branin_hoo(x))
# %% BO
if __name__ == "__main__":  # confirms that the code is under main function
    # Iteration parameters
    n_loop = 10
    # BO parameters
    n_max_iter = 50
    n_init_samples=5
    n_cand=10
    n_worker=10
    n_mcmc=30
    pbounds = {'x1':(-5, 10), 'x2':(0, 15)}

    consider_time = True
    kernel_fun = 'matern' # 'matern' / 'se'
    use_pending = True
    minus_obj_fun = True
    best_each_loop = run_bo(obj, n_loop, pbounds, n_init_samples, n_worker, n_cand, n_mcmc, n_max_iter, consider_time, kernel_fun, use_pending, 
    minus_obj_fun = True)
    np.savetxt('result/prob1_bo.csv', best_each_loop, delimiter=",")

    # %% Neldear-Mead
    # Get random initial point to run optimization
    s_opt = Samples(obj, bounds=pbounds, n_init_samples=n_loop, n_worker=n_worker, n_cand=n_cand, n_mcmc=n_mcmc)
    s_opt.random_sampling(n_loop, init_sampling=True)

    # Run optimization in each loop
    best_each_loop = []
    for x0 in s_opt.X0:
        best_single_loop = []
        result = minimize(branin_hoo, x0, method='Nelder-Mead', bounds=s_opt.bounds, callback=callback, options = {'fatol':1e-20})
        best_each_loop.append(best_single_loop)
    best_each_loop = [[sublist[i] if i < len(sublist) else sublist[-1] for i in range(50)] for sublist in best_each_loop]
    best_each_loop = np.array(best_each_loop)
    np.savetxt('result/prob1_opt.csv', best_each_loop, delimiter=",")

    #%% Random search
    best_each_loop = []
    for _ in range(n_loop):
        s_rand = Samples(obj, bounds=pbounds, n_init_samples=n_init_samples, n_worker=n_worker, n_cand=n_cand, n_mcmc=n_mcmc)
        s_rand.random_sampling(n_max_iter, init_sampling=True)
        y = branin_hoo_2(s_rand.X0)
        best_single_loop = np.zeros_like(y)
        best_single_loop[0] = y[0] 
        for i in range(1, len(y)):
            best_single_loop[i] = min(best_single_loop[i-1], y[i]) # The current value in b is the minimum of the previous value in b and the current value in a
        best_each_loop.append(best_single_loop)

    np.savetxt('result/prob1_rnd.csv', best_each_loop, delimiter=",")