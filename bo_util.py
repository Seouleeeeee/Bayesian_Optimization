from multiprocessing import Process, Queue
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats import norm
import time 
import warnings



# %%
class Samples(object):
    def __init__(self, obj_func, bounds=None, n_init_samples=100, n_cand=10, n_worker=10, n_mcmc=20, prob3=False):
        self.obj_func = obj_func
        self.n_init_samples = n_init_samples
        self.n_cand = n_cand
        self.n_mcmc = n_mcmc
        self.n_worker = n_worker
        self.prob3 = prob3
        if self.prob3==True:
            self.bounds = list(range(6))
        elif type(bounds) is dict:
            self.bounds = [val for val in bounds.values()]
        else:
            self.bounds = bounds

    def random_sampling(self, n_points, init_sampling=False):
        def rand_select(bounds, n_points):
            if self.prob3:
                r = np.vstack((
                    np.random.choice(np.arange(16, 128, 16), (1, n_points)),
                    np.random.choice(np.arange(16, 128, 16), (1, n_points)),
                    np.random.choice(np.arange(16, 128, 16), (1, n_points)),
                    np.random.choice([0, 1, 2], (1, n_points)),
                    np.random.choice(np.arange(16, 256, 16), (1, n_points)),
                    np.random.choice(np.arange(0.1, 0.6, 0.1), (1, n_points))
                ))
                return r
            else:
                return [np.random.uniform(prange[0], prange[1], n_points) for _, prange in enumerate(bounds)]
        

        def check_dist_close(X0, X1):
            dist = cdist(X0, X1, 'sqeuclidean')
            dist[np.tril_indices(len(X0), 0, len(X1))]=np.nan # Make lower traingular + diagnoal elements as Nan
            idx_too_close_X1, idx_too_close_X1 = np.where(dist<1e-06)
            return idx_too_close_X1

        def change_vals_too_close(self, X0, X1):
            dist = cdist(X0, X1, 'sqeuclidean')
            dist[np.tril_indices(len(dist))]=np.nan # Make lower traingular + diagnoal elements as Nan
            _, idx_too_close_X1 = np.where(dist<0.01)
            new_elems_X1 = rand_select(self.bounds, len(idx_too_close_X1))
            X1[idx_too_close_X1, :] = np.asarray(new_elems_X1).reshape(len(self.bounds), -1).T
            return X1
            
        ## Avoid sampling same or too closed points
        if init_sampling: # Initial sampling
            rand_x = rand_select(self.bounds, n_points)
            X0=np.asarray(rand_x).T
            idx_too_close = check_dist_close(X0, X0)
            while idx_too_close.size>0:
                X0 = change_vals_too_close(self, X0, X0)
                idx_too_close = check_dist_close(X0, X0)
            
            self.X0=X0
            self.y0, self.duration =self.obj_func(self.X0)
            self.y0 = np.asarray(self.y0)
            self.duration = np.asarray(self.duration)
        else:
            rand_x = rand_select(self.bounds, n_points)
            # Delete too close sample in itself
            X1=np.asarray(rand_x).T
            idx_too_close = check_dist_close(X1, X1)
            while idx_too_close.size>0:
                X1 = change_vals_too_close(self, X1, X1)
                idx_too_close = check_dist_close(X1, X1)
            # Delete too close sample from X0
            idx_too_close = check_dist_close(self.X0, X1)
            while idx_too_close.size>0:
                X1 = change_vals_too_close(self, self.X0, X1)
                idx_too_close = check_dist_close(self.X0, X1)

            self.X_cp = np.array_split(X1, self.n_worker)
            # Divide for each worker

    def add_samples(self, X_elected):
        y_elected, time_elected = self.obj_func(X_elected)
        self.X0 = np.vstack((self.X0, X_elected))
        self.y0 = np.concatenate((self.y0, y_elected))
        self.duration = np.concatenate((self.duration, time_elected))

    def max_including_previous(self):
        # Get maximum values including previous
        b = np.zeros_like(self.y0)
        b[0] = self.y0[0] 
        for i in range(1, len(self.y0)):
            b[i] = max(b[i-1], self.y0[i]) # The current value in b is the minimum of the previous value in b and the current value in a
        return b


# %%
# Calculate GP hyperparameters theta_0 with X_0, and y_0

class GaussianProcess(Samples):
    def __init__(self, samples, kernel, obs_noise=0):
        super().__init__(samples.obj_func, samples.bounds)
        self.kernel = kernel
        self.obs_noise = obs_noise
        self.X0 = samples.X0
        self.y0 = samples.y0
        self.duration = samples.duration
        self.n_mcmc = samples.n_mcmc
    
    def maximize_log_marginal_likelihood(self, variance=1, length=1, save_params=False):
        bounds = [(1e-5, 1e5), (1e-5, 1e5)]
        opt_result = minimize(minus_log_marginal_likelihood, [variance, length], args=(self.X0, self.y0, self.kernel), bounds=bounds)
        if save_params==True:
            self.theta = opt_result.x
        else:
            return opt_result.x
        
    def maximize_log_marginal_likelihood_time(self, variance=1, length=1):
        bounds = [(1e-5, 1e5), (1e-5, 1e5)]
        opt_result = minimize(minus_log_marginal_likelihood, [variance, length], args=(self.X0, self.duration, self.kernel), bounds=bounds)
        self.theta_time = opt_result.x

    def generate_mcmc_hypers(self, target='y'):
        if target == 'y':
            params_mean = self.theta
            params_range= params_mean/2
            params_lb = params_mean.reshape(1, -1)-params_range.reshape(1, -1)
            params_lb[params_lb<=0] = 1e-3
            params_ub = params_mean.reshape(1, -1)+params_range.reshape(1, -1)
            params_mcmc = np.random.uniform(params_lb, params_ub, size = (self.n_mcmc, params_mean.shape[0]))
            self.params_mcmc = params_mcmc
        elif target == 'time':
            params_mean = self.theta_time
            params_range= params_mean/2
            params_lb = params_mean.reshape(1, -1)-params_range.reshape(1, -1)
            params_lb[params_lb<=0] = 1e-3
            params_ub = params_mean.reshape(1, -1)+params_range.reshape(1, -1)
            params_mcmc = np.random.uniform(params_lb, params_ub, size = (self.n_mcmc, params_mean.shape[0]))
            self.params_mcmc = params_mcmc

    def predict(self, X1):
        variance, length = self.theta
        K_00 = self.kernel(self.X0, self.X0, variance, length)
        K_00[np.diag_indices_from(K_00)] += 1e-5
        L_00 = cholesky(K_00, lower=True, check_finite=False)
        alpha = cho_solve((L_00, True), self.y0)
        K_10 = self.kernel(X1, self.X0, variance, length)
        pos_mean = K_10@alpha

        K_11 = self.kernel(X1, X1, variance, length)
        beta = cho_solve((L_00, True), K_10.T)
        pos_var = K_11 - K_10@beta

        return pos_mean, pos_var

def minus_log_marginal_likelihood(theta, X0, y0, kernel):
    variance, length = theta
    K_00 = kernel(X0, X0, variance, length)
    K_00[np.diag_indices_from(K_00)] += 1e-5
    L_00 = cholesky(K_00, lower=True, check_finite=False)
    alpha = cho_solve((L_00, True), y0)
    lml= -0.5*(y0.T)@alpha - np.log(np.diag(L_00)).sum() - 0.5*L_00.shape[0]*np.log(2*np.pi)
    return -lml



def squared_exponential(x1, x2, variance, length):
    dist = cdist(x1/length, x2/length, 'sqeuclidean')
    return variance * np.exp(-0.5*dist)

def matern_25(X, Y, variance, length_scale):
    dists = cdist(X / length_scale, Y / length_scale, metric="euclidean")
    K = dists * np.sqrt(5)
    K = variance*(1.0 + K + K**2 / 3.0) * np.exp(-K)
    return K


# %%
## Get more plausible candidate points - Find good points among cand
# 0: complete set
# c: candidate
# p: pending
# f: complete + pending (fantasies)

# mu: mean function
# K: kernel matrix
# L: cholesky of k

def calc_ei_over_hypers(s, gp, X_cand, X_pend, consider_time):
    ei_cand_over_hypers = np.zeros((X_cand.shape[0], s.n_mcmc))
    if X_pend is not None:
        # Iterate over hyperparameters
        for i, (variance, length) in enumerate(gp.params_mcmc):
            # Get surrogate model values of X_p
            K_00 = gp.kernel(s.X0, s.X0, variance, length)
            K_00[np.diag_indices_from(K_00)] += 1e-10
            L_00 = cholesky(K_00, lower=True)
            alpha = cho_solve((L_00, True), s.y0)
            K_p0 = gp.kernel(X_pend, s.X0, variance, length)
            y_pend_pred = K_p0@alpha

            # Combine fantasies set
            X_f = np.vstack((s.X0, X_pend))
            y_f = np.concatenate((s.y0, y_pend_pred))

            # Get GP posterior of X_c using X_f as training set
            K_ff = gp.kernel(X_f, X_f, variance, length)
            K_ff[np.diag_indices_from(K_ff)] += 1e-10
            L_ff = cholesky(K_ff, lower=True)
            alpha = cho_solve((L_ff, True), y_f)
            K_c0 = gp.kernel(X_cand, X_f, variance, length)
            mu_c = K_c0@alpha

            beta = cho_solve((L_ff, True), K_c0.T)
            K_cc = gp.kernel(X_cand, X_cand, variance, length)
            var_c = K_cc -K_c0@beta
            sigma_c = np.sqrt(np.abs(np.diag(var_c)))

            # Calculate EI of candidates
            y_best = max(y_f)
            Z = (mu_c-y_best)/sigma_c
            EI_cand = (mu_c-y_best)*norm.cdf(Z) + sigma_c*norm.pdf(Z)

            # Save
            if consider_time:
                alpha_time = cho_solve((L_00, True), s.duration)
                K_c0 = gp.kernel(X_cand, s.X0, variance, length)
                time_pred = K_c0@alpha_time
                time_pred[time_pred<0.01*np.mean(s.duration)] = 0.1*np.mean(s.duration)
                ei_cand_over_hypers[:, i] = EI_cand/time_pred
            else:
                ei_cand_over_hypers[:, i] = EI_cand
            # Modeling time
    else:
        # Iterate over hyperparameters
        for i, (variance, length) in enumerate(gp.params_mcmc):
            # Get surrogate model values of X_p
            K_00 = gp.kernel(s.X0, s.X0, variance, length)
            K_00[np.diag_indices_from(K_00)] += 1e-10
            L_00 = cholesky(K_00, lower=True)
            alpha = cho_solve((L_00, True), s.y0)
            K_c0 = gp.kernel(X_cand, s.X0, variance, length)
            mu_c = K_c0@alpha

            beta = cho_solve((L_00, True), K_c0.T)
            K_cc = gp.kernel(X_cand, X_cand, variance, length)
            var_c = K_cc -K_c0@beta
            sigma_c = np.sqrt(np.abs(np.diag(var_c)))

            # Calculate EI of candidates
            y_best = max(s.y0)
            Z = (mu_c-y_best)/sigma_c
            EI_cand = (mu_c-y_best)*norm.cdf(Z) + sigma_c*norm.pdf(Z)

            # Save
            if consider_time:
                alpha_time = cho_solve((L_00, True), s.duration)
                K_c0 = gp.kernel(X_cand, s.X0, variance, length)
                time_pred = K_c0@alpha_time
                time_pred[time_pred<0.01*np.mean(s.duration)] = 0.1*np.mean(s.duration)
                ei_cand_over_hypers[:, i] = EI_cand/time_pred
            else:
                ei_cand_over_hypers[:, i] = EI_cand
            # Modeling time
        
        
    return ei_cand_over_hypers


# %%
## Get more plausible candidate points - Get optimal points and add them to cand points

def calc_ei_one_cand(X_cand, s, variance, length, kernel, consider_time):
    K_00 = kernel(s.X0, s.X0, variance, length)
    K_00[np.diag_indices_from(K_00)] += 1e-10
    L_00 = cholesky(K_00, lower=True)
    alpha = cho_solve((L_00, True), s.y0)
    K_c0 = kernel(X_cand, s.X0, variance, length)
    mu_c = K_c0@alpha

    beta = cho_solve((L_00, True), K_c0.T)
    K_cc = kernel(X_cand, X_cand, variance, length)
    var_c = K_cc -K_c0@beta
    sigma_c = np.sqrt(np.abs(np.diag(var_c)))

    y_best = max(s.y0)
    Z = (mu_c-y_best)/sigma_c
    EI_cand = (mu_c-y_best)*norm.cdf(Z) + sigma_c*norm.pdf(Z)

    if consider_time:
        alpha_time = cho_solve((L_00, True), s.duration)
        time_pred = K_c0@alpha_time
        EI_cand /= time_pred
    return EI_cand

def sum_ei_over_hyper(X_cand, s, kernel, params_mcmc, consider_time):
    summed_ei_over_hypers=0
    X_cand = X_cand.reshape(-1, len(s.bounds))
    if np.isnan(X_cand).any():
        return 1000
    for variance, length in params_mcmc:
        # ei = calc_ei_one_cand(X_cand.reshape(-1, 2), s, variance, length, kernel, consider_time)
        K_00 = kernel(s.X0, s.X0, variance, length)
        K_00[np.diag_indices_from(K_00)] += 1e-10
        L_00 = cholesky(K_00, lower=True)
        alpha = cho_solve((L_00, True), s.y0)
        K_c0 = kernel(X_cand, s.X0, variance, length)
        mu_c = K_c0@alpha

        beta = cho_solve((L_00, True), K_c0.T)
        K_cc = kernel(X_cand, X_cand, variance, length)
        var_c = K_cc -K_c0@beta
        sigma_c = np.sqrt(np.abs(np.diag(var_c)))

        y_best = max(s.y0)
        Z = (mu_c-y_best)/sigma_c
        ei = (mu_c-y_best)*norm.cdf(Z) + sigma_c*norm.pdf(Z)

        if consider_time:
            alpha_time = cho_solve((L_00, True), s.duration)
            time_pred = K_c0@alpha_time
            if time_pred < np.mean(s.duration)*0.01:
                time_pred = 0.1*np.mean(s.duration)
            ei /= time_pred
        summed_ei_over_hypers+=ei
    return -summed_ei_over_hypers


#%%
def assign_work(s, current_worker, gp, run_opt=False, consider_time=True, use_pending=True):
    X_cand = s.X_cp[current_worker]
    X_pend = [x for i, x in enumerate(s.X_cp) if i!=current_worker]
    X_pend = np.vstack(X_pend)

    # Choose best candidate points that will include in the complete set
    if use_pending:
        ei_cand_over_hypers = calc_ei_over_hypers(s, gp, X_cand, X_pend, consider_time=consider_time)
    else:
        X_pend = None
        ei_cand_over_hypers = calc_ei_over_hypers(s, gp, X_cand, X_pend, consider_time=consider_time)
    ei_cand_mean = np.mean(ei_cand_over_hypers, axis=1)
    
    if run_opt: ## Use top 20% candidates as the initial point of optimization
        idxs = np.argsort(ei_cand_mean)[-int(s.n_cand*0.2):] 
        X_opt_init = X_cand[idxs, :]
        X_cand_optimum = []
        for X_init in X_opt_init:
            opt_res = minimize(sum_ei_over_hyper, X_init, args=(s, gp.kernel, gp.params_mcmc, consider_time), bounds=s.bounds)
            X_cand_optimum.append(opt_res.x)
        ## Add the result of optimization to candidate set
        X_cand = np.vstack((X_cand, np.asarray(X_cand_optimum)))
        ei_cand_over_hypers = calc_ei_over_hypers(s, gp, X_cand, X_pend, consider_time=consider_time)
        ## Choose the best candidate and evaluate with the objective function
        ei_cand_mean = np.mean(ei_cand_over_hypers, axis=1)

    best_idx=np.argmax(ei_cand_mean)
    elected_cand = X_cand[best_idx, :]
    return elected_cand

def worker(s, current_worker, gp, queue, consider_time=True):
    run_opt=False
    elected_cand = assign_work(s, current_worker, gp, run_opt, consider_time)
    queue.put(elected_cand)

def run_bo(obj, n_loop, pbounds, n_init_samples, n_worker, n_cand, n_mcmc, n_max_iter, consider_time, kernel_fun, use_pending, minus_obj_fun, prob3=False):
    best_each_loop = []
    period_each_loop = []
    for _ in range(n_loop):
        running_period = []
        # Generate complete set X_0, y_0 with the size of n_init from random sampling
        best_results = []
        s = Samples(obj, bounds=pbounds, n_init_samples=n_init_samples, n_worker=n_worker, n_cand=n_cand, n_mcmc=n_mcmc, prob3=prob3)
        s.random_sampling(n_init_samples, init_sampling=True)
        best_results.append(max(s.y0))

        # Choose kernel function
        if kernel_fun == 'matern':
            kernel = matern_25
        else:
            kernel = squared_exponential

        for n_iter in range(n_max_iter):
            start_time = time.time()
            # Compute hyperparameters that maximizing log-likelihood of (X, y)
            gp = GaussianProcess(s, kernel) 
            gp.maximize_log_marginal_likelihood(save_params=True)
            # Compute hyperparameters that maximizing log-likelihood (X, time)
            gp.maximize_log_marginal_likelihood_time()

            # Generate n_parallel candidate sets (candidate & pending)
            s.random_sampling(n_cand*n_worker)

            # Generate n_mcmc new GP hyperparameters of (X, y)
            gp.generate_mcmc_hypers(target='y')
            gp.generate_mcmc_hypers(target='time')
            
            # Get best candidate among n_worker
            # Using concept of fantasies, but not directly using parallel computing owing to computing resources and generalization
            elected_cands = [] 
            for current_worker in range(n_worker):
                elected_cand = assign_work(s, current_worker, gp, False, consider_time, use_pending)
                elected_cands.append(elected_cand)
            predicted_cands, _ = gp.predict(elected_cands)
            cand_best = elected_cands[np.argmax(predicted_cands)]
            
            # Add best candidate and evaluate with real objective function
            s.add_samples(np.array(cand_best))
            best_results.append(max(s.y0))
            end_time = time.time()
            one_loop_period=end_time-start_time
            running_period.append(one_loop_period)

        best_single_loop = s.max_including_previous()
        if minus_obj_fun:
            best_each_loop.append(-best_single_loop)
        else:
            best_each_loop.append(best_single_loop)
        period_each_loop.append(running_period)

    if prob3 == True:
        return best_each_loop, period_each_loop
    else:
        return best_each_loop