import jax
import jax.numpy as jnp
from jax import grad, jit
import numpy as np
from scipy.optimize import minimize


class ComparativeAdvantageModel:

    def __init__(self, n_factors):
        self.n_factors = n_factors
        self.fitted_params = None
        self.iteration = 0

    def log_sigmoid(self, x):
        return jnp.where(
            x >= 0,
            -jnp.log1p(jnp.exp(-x)),
            x - jnp.log1p(jnp.exp(x))
        )

    def neg_log_likelihood(self, params, Y, weights=None):
        """
        This is your core negative log-likelihood, in JAX. 
        It may print iteration info,
        but note that if you jit this function, printing may not occur every call.
        """
        country_intercepts = params[: self.num_countries]
        product_intercepts = params[self.num_countries : self.num_countries + self.num_products]
        
        # factor vectors
        country_vectors_flat = params[
            self.num_countries + self.num_products :
            self.num_countries + self.num_products + self.num_countries * self.n_factors
        ]
        product_vectors_flat = params[
            self.num_countries + self.num_products + self.num_countries * self.n_factors :
        ]
        
        country_vectors = country_vectors_flat.reshape(self.num_countries, self.n_factors)
        product_vectors = product_vectors_flat.reshape(self.num_products, self.n_factors)
        
        X = country_intercepts[:, jnp.newaxis] + product_intercepts + (country_vectors @ product_vectors.T)
        
        log_likelihood = (
            Y * self.log_sigmoid(X)
            + (1 - Y) * self.log_sigmoid(-X)
        )
        
        if weights is not None:
            log_likelihood = log_likelihood * weights
        
        # You can keep a manual iteration counter here.
        # call(lambda x: print(f'Iteration {self.iteration}, log_likelihood: {x:.2f}'), jnp.sum(log_likelihood))
        self.iteration += 1
        
        return -jnp.sum(log_likelihood)

    def fit(self, Y, weights=None, method='L-BFGS-B', max_iter=150000, max_fun=20000):
        # Convert data to jax arrays
        Y = jnp.array(Y)
        if weights is not None:
            weights = jnp.array(weights)
            
        self.num_countries, self.num_products = Y.shape
        # total_params = intercepts for countries + intercepts for products + 
        # factor vectors for countries + factor vectors for products
        self.total_params = (
            self.num_countries + 
            self.num_products + 
            self.num_countries * self.n_factors + 
            self.num_products  * self.n_factors
        )
        
        self.iteration = 0
        
        # Wrap your JAX nll in a Python function that returns float
        # so that scipy can handle it
        def objective(params_np):
            """
            params_np is a NumPy array (float64).
            Convert to jax.numpy array, call self.neg_log_likelihood,
            return a float so that SciPy is happy.
            """
            params_jnp = jnp.array(params_np)
            val_jnp = self.neg_log_likelihood(params_jnp, Y, weights)
            return val_jnp
        
        # JIT only the gradient for speed, but not the objective
        # so that printing inside neg_log_likelihood is still visible.
        grad_neg_log_likelihood = jit(grad(self.neg_log_likelihood, argnums=0))
        
        def objective_grad(params_np):
            """
            Return the gradient as a NumPy array so that SciPy is happy.
            """
            params_jnp = jnp.array(params_np)
            grad_jnp = grad_neg_log_likelihood(params_jnp, Y, weights)
            return np.array(grad_jnp, dtype=np.float64)
        
        # Create an initial guess as a NumPy array
        key = jax.random.PRNGKey(42)
        x0_jax = jax.random.normal(key, (self.total_params,))
        x0 = np.array(x0_jax, dtype=np.float64)
        
        # Now do the actual minimize
        result = minimize(
            fun=objective,
            x0=x0,
            jac=objective_grad,
            method=method,
            options={'maxiter': max_iter, 'maxfun': max_fun}
        )
        
        print(f'Success: {result.success}, Message: {result.message}')
        
        # Extract the fitted parameter values
        self.fitted_params = result.x
        
        # Unpack learned parameters into interpretable shapes
        self.country_intercepts = self.fitted_params[: self.num_countries]
        self.product_intercepts = self.fitted_params[self.num_countries : self.num_countries + self.num_products]
        idx = self.num_countries + self.num_products
        
        country_vecs_flat = self.fitted_params[idx : idx + self.num_countries*self.n_factors]
        idx += self.num_countries*self.n_factors
        product_vecs_flat = self.fitted_params[idx : idx + self.num_products*self.n_factors]
        
        self.country_vectors = country_vecs_flat.reshape(self.num_countries, self.n_factors)
        self.product_vectors = product_vecs_flat.reshape(self.num_products, self.n_factors)
        
        return {
            'country_intercepts': self.country_intercepts,
            'product_intercepts': self.product_intercepts,
            'country_vectors': self.country_vectors,
            'product_vectors': self.product_vectors
        }

    def predict_proba(self):
        if self.fitted_params is None:
            raise ValueError("Модель еще не обучена. Сначала вызовите метод fit().")
        
        X_fitted = (
            self.country_intercepts[:, jnp.newaxis] 
            + self.product_intercepts 
            + (self.country_vectors @ self.product_vectors.T)
        )
        return jax.nn.sigmoid(X_fitted)
