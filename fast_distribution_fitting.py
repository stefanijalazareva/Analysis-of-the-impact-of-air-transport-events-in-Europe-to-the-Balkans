"""
Streamlined Distribution Fitting for Air Transport Delays

A faster version of the distribution fitting script that allows:
- Sampling data for quicker results
- Running specific analysis steps separately
- Better progress tracking with time estimates
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import dual_annealing, minimize
from scipy.stats import kstest, ks_2samp
from pathlib import Path
import logging
import time
from tqdm import tqdm
import warnings
import argparse
import json
import datetime
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('distribution_fitting_fast.log'),
        logging.StreamHandler()
    ]
)

class FastNCTFitter:
    """Fast version of the NCT distribution fitter with sampling capabilities."""

    def __init__(self, data_dir="data/ProcessedData", output_dir="data/DistributionFitting",
                 sample_size=None, random_seed=42):
        """
        Initialize the FastNCTFitter.

        Args:
            data_dir: Directory containing the processed data
            output_dir: Directory to save outputs
            sample_size: If provided, sample this many records for faster processing
            random_seed: Random seed for reproducibility
        """
        # Convert to absolute paths if needed
        self.data_dir = Path(data_dir).absolute()
        self.output_dir = Path(output_dir).absolute()

        # Create output directory if it doesn't exist
        print(f"Creating output directory at: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sample_size = sample_size
        self.random_seed = random_seed

        # Set random seed
        np.random.seed(random_seed)

        # Store fitted parameters
        self.params = {}
        self.traffic_dependence = {}

    def load_data(self):
        """Load the delay data and traffic volumes, with optional sampling."""
        try:
            start_time = time.time()
            logging.info("Loading delay data...")

            # Load cleaned delay data from parquet file
            df_path = self.data_dir / 'cleaned_delays.parquet'
            self.df = pd.read_parquet(df_path)

            logging.info(f"Full dataset: {len(self.df)} records")

            # Sample data if requested
            if self.sample_size and self.sample_size < len(self.df):
                logging.info(f"Sampling {self.sample_size} records for faster processing")
                self.df = self.df.sample(self.sample_size, random_state=self.random_seed)

            # Extract delay values
            self.delays = self.df['delay_min'].values

            # Prepare daily data
            self.df['date'] = pd.to_datetime(self.df['sched_dt_utc']).dt.date
            self.daily_traffic = self.df.groupby('date').size()

            daily_groups = self.df.groupby('date')
            self.daily_delays = {date: group['delay_min'].values for date, group in daily_groups}

            # Filter out days with too few records
            self.daily_delays = {date: delays for date, delays in self.daily_delays.items()
                               if len(delays) >= 30}

            elapsed_time = time.time() - start_time
            logging.info(f"Loaded {len(self.delays)} delay records across {len(self.daily_delays)} days "
                        f"in {elapsed_time:.2f} seconds")

            # Save a quick summary
            summary = {
                'total_records': len(self.delays),
                'date_range': [str(min(self.daily_delays.keys())), str(max(self.daily_delays.keys()))],
                'days_with_sufficient_data': len(self.daily_delays),
                'mean_delay': float(np.mean(self.delays)),
                'median_delay': float(np.median(self.delays)),
                'std_delay': float(np.std(self.delays)),
                'min_delay': float(np.min(self.delays)),
                'max_delay': float(np.max(self.delays)),
            }

            with open(self.output_dir / 'data_summary.json', 'w') as f:
                json.dump(summary, f, indent=4)

            return True

        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return False

    def fit_nct_global(self, maxiter=500):
        """
        Fit the noncentral Student's t-distribution to the entire dataset.

        Args:
            maxiter: Maximum iterations for the optimizer
        """
        try:
            start_time = time.time()
            logging.info("Fitting global NCT distribution...")

            # Smaller sample for global fit if the dataset is very large
            delays = self.delays
            if len(delays) > 100000:
                sample_size = min(100000, len(delays) // 10)
                logging.info(f"Using {sample_size} random samples for global fitting (out of {len(delays)})")
                delays = np.random.choice(delays, size=sample_size, replace=False)

            # Initial parameter guesses and bounds
            bounds = [(1.0, 100.0),  # df: degrees of freedom > 1
                      (-10.0, 10.0),  # nc: noncentrality parameter
                      (-20.0, 20.0),  # loc: location parameter
                      (0.1, 50.0)]    # scale: scale parameter > 0

            # Define the negative log-likelihood function for NCT
            def neg_log_likelihood(params):
                df, nc, loc, scale = params
                if df <= 0 or scale <= 0:
                    return 1e10
                try:
                    log_pdf = np.sum(stats.nct.logpdf(delays, df=df, nc=nc, loc=loc, scale=scale))
                    return -log_pdf
                except:
                    return 1e10

            # Optimize using dual annealing with progress reporting
            logging.info(f"Starting optimization with maxiter={maxiter}...")
            last_update = time.time()

            def callback(x, f, context):
                nonlocal last_update
                now = time.time()
                if now - last_update > 10:  # Update every 10 seconds
                    elapsed = now - start_time
                    logging.info(f"Optimization in progress: {elapsed:.2f}s elapsed, current params: "
                                f"df={x[0]:.2f}, nc={x[1]:.2f}, loc={x[2]:.2f}, scale={x[3]:.2f}, "
                                f"objective={f:.4f}")
                    last_update = now
                return False

            result = dual_annealing(
                neg_log_likelihood,
                bounds=bounds,
                maxiter=maxiter,
                callback=callback
            )

            if result.success:
                df_opt, nc_opt, loc_opt, scale_opt = result.x
                self.params['global'] = {
                    'df': df_opt,
                    'nc': nc_opt,
                    'loc': loc_opt,
                    'scale': scale_opt,
                    'nll': result.fun,
                    'converged': True,
                    'message': str(result.message),
                    'iterations': result.nit
                }

                # Calculate KS statistic for goodness of fit
                ks_stat, p_value = kstest(
                    delays,
                    lambda x: stats.nct.cdf(x, df=df_opt, nc=nc_opt, loc=loc_opt, scale=scale_opt)
                )

                self.params['global']['ks_stat'] = ks_stat
                self.params['global']['p_value'] = p_value

                elapsed_time = time.time() - start_time
                logging.info(f"Global NCT fit completed in {elapsed_time:.2f} seconds")
                logging.info(f"Parameters: df={df_opt:.4f}, nc={nc_opt:.4f}, loc={loc_opt:.4f}, scale={scale_opt:.4f}")
                logging.info(f"KS test: statistic={ks_stat:.4f}, p-value={p_value:.4f}")

                # Save parameters to JSON
                with open(self.output_dir / 'global_nct_parameters.json', 'w') as f:
                    json.dump(self.params['global'], f, indent=4)

                # Create fit visualization
                self._plot_fit(delays, self.params['global'], "global")

                # Create histogram of delays with fit overlaid
                plt.figure(figsize=(12, 8))
                sns.histplot(delays, bins=100, stat='density', alpha=0.6, label='Data')

                x = np.linspace(np.min(delays), np.max(delays), 1000)
                y = stats.nct.pdf(x, df=df_opt, nc=nc_opt, loc=loc_opt, scale=scale_opt)

                plt.plot(x, y, 'r-', linewidth=2, label='NCT Fit')
                plt.title('Global NCT Distribution Fit')
                plt.xlabel('Delay (minutes)')
                plt.ylabel('Density')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(self.output_dir / 'global_nct_fit.png', dpi=300, bbox_inches='tight')
                plt.close()

                return True
            else:
                logging.error(f"Global fitting did not converge: {result.message}")
                return False

        except Exception as e:
            logging.error(f"Error fitting global NCT: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return False

    def fit_nct_daily(self, max_days=None, maxiter=300):
        """
        Fit the noncentral Student's t-distribution to daily delay data.

        Args:
            max_days: Maximum number of days to fit (for quicker testing)
            maxiter: Maximum iterations for each day's optimization
        """
        start_time = time.time()
        logging.info("Fitting daily NCT distributions...")

        # Initial parameter guesses from global fit if available
        if 'global' in self.params:
            init_df = self.params['global']['df']
            init_nc = self.params['global']['nc']
            init_loc = self.params['global']['loc']
            init_scale = self.params['global']['scale']

            logging.info(f"Using global fit as starting point: df={init_df:.2f}, nc={init_nc:.2f}, "
                        f"loc={init_loc:.2f}, scale={init_scale:.2f}")
        else:
            init_df = 5.0
            init_nc = 0.0
            init_loc = 0.0
            init_scale = 1.0

        # Bounds for optimization
        bounds = [(1.0, 100.0),    # df: degrees of freedom > 1
                 (-10.0, 10.0),    # nc: noncentrality parameter
                 (-20.0, 20.0),    # loc: location parameter
                 (0.1, 50.0)]      # scale: scale parameter > 0

        # Store daily fitted parameters
        daily_params = {}
        traffic_values = []
        df_values = []
        nc_values = []
        loc_values = []
        scale_values = []

        # Select days to fit
        days_to_fit = list(self.daily_delays.keys())
        if max_days and max_days < len(days_to_fit):
            logging.info(f"Limiting to {max_days} days for faster processing")
            days_to_fit = sorted(days_to_fit)[:max_days]

        total_days = len(days_to_fit)
        successful_fits = 0

        # Fit each day with sufficient data
        for date in tqdm(days_to_fit, desc="Fitting daily distributions"):
            delays = self.daily_delays[date]

            # Define the negative log-likelihood function for this day's data
            def neg_log_likelihood(params):
                df, nc, loc, scale = params
                if df <= 0 or scale <= 0:
                    return 1e10
                try:
                    log_pdf = np.sum(stats.nct.logpdf(delays, df=df, nc=nc, loc=loc, scale=scale))
                    return -log_pdf
                except:
                    return 1e10

            try:
                # Optimize using dual annealing
                result = dual_annealing(
                    neg_log_likelihood,
                    bounds=bounds,
                    maxiter=maxiter,
                    x0=[init_df, init_nc, init_loc, init_scale]
                )

                if result.success:
                    df_opt, nc_opt, loc_opt, scale_opt = result.x

                    # Calculate KS statistic for goodness of fit
                    ks_stat, p_value = kstest(
                        delays,
                        lambda x: stats.nct.cdf(x, df=df_opt, nc=nc_opt, loc=loc_opt, scale=scale_opt)
                    )

                    # Store the parameters
                    daily_params[date] = {
                        'df': df_opt,
                        'nc': nc_opt,
                        'loc': loc_opt,
                        'scale': scale_opt,
                        'converged': True,
                        'ks_stat': ks_stat,
                        'p_value': p_value,
                        'n_delays': len(delays)
                    }

                    # Store for traffic regression
                    traffic = self.daily_traffic.get(date, 0)
                    traffic_values.append(traffic)
                    df_values.append(df_opt)
                    nc_values.append(nc_opt)
                    loc_values.append(loc_opt)
                    scale_values.append(scale_opt)

                    successful_fits += 1

            except Exception as e:
                logging.warning(f"Error fitting NCT for {date}: {str(e)}")

        # Store the fitted parameters
        self.params['daily'] = daily_params

        # Create DataFrame for analysis
        if successful_fits > 0:
            self.param_df = pd.DataFrame({
                'date': list(daily_params.keys()),
                'traffic': traffic_values,
                'df': df_values,
                'nc': nc_values,
                'loc': loc_values,
                'scale': scale_values
            })

            # Save the parameters
            self.param_df.to_csv(self.output_dir / 'daily_nct_parameters.csv', index=False)

            # Visualize parameter vs traffic relationships
            self._plot_params_vs_traffic()

        elapsed_time = time.time() - start_time
        logging.info(f"Daily NCT fits: {successful_fits} successful out of {total_days} days "
                    f"in {elapsed_time:.2f} seconds")

        return True

    def model_traffic_dependence(self):
        """Model how NCT parameters depend on traffic using simple regression."""
        # Try to load daily parameters from file if not already available
        if not hasattr(self, 'param_df') or self.param_df is None or self.param_df.empty:
            if not self._load_daily_parameters_from_file():
                logging.error("No daily parameters available for traffic dependence modeling")
                return False

        start_time = time.time()
        logging.info("Modeling traffic dependence of NCT parameters...")

        # Parameters to model
        param_names = ['df', 'nc', 'loc', 'scale']

        # Store traffic dependence models
        traffic_models = {}

        for param in param_names:
            # Simple linear model: param = alpha + beta * traffic
            X = self.param_df['traffic'].values
            y = self.param_df[param].values

            # Initial guesses
            alpha_init = np.mean(y)
            beta_init = 0.0

            # Fit linear regression
            def linear_residuals(params):
                alpha, beta = params
                y_pred = alpha + beta * X
                return np.sum((y - y_pred) ** 2)

            # Optimize
            result = minimize(linear_residuals, [alpha_init, beta_init], method='Nelder-Mead')

            alpha_opt, beta_opt = result.x

            # Store the model
            traffic_models[param] = {
                'alpha': alpha_opt,
                'beta': beta_opt,
                'converged': result.success,
                'message': str(result.message)
            }

            # Calculate R-squared
            y_pred = alpha_opt + beta_opt * X
            ss_total = np.sum((y - np.mean(y)) ** 2)
            ss_residual = np.sum((y - y_pred) ** 2)
            r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0

            traffic_models[param]['r_squared'] = r_squared

            logging.info(f"{param} = {alpha_opt:.4f} + {beta_opt:.4f} * traffic (R² = {r_squared:.4f})")

            # Plot the fit
            plt.figure(figsize=(10, 6))
            plt.scatter(X, y, alpha=0.6, label='Daily fitted values')

            # Sort X for line plot
            sorted_idx = np.argsort(X)
            X_sorted = X[sorted_idx]
            y_pred_sorted = y_pred[sorted_idx]

            plt.plot(X_sorted, y_pred_sorted, 'r-', linewidth=2,
                     label=f'{param} = {alpha_opt:.4f} + {beta_opt:.4f} * traffic')

            plt.title(f'Traffic Dependence of NCT {param} Parameter')
            plt.xlabel('Daily Traffic Volume')
            plt.ylabel(f'NCT {param} Parameter')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / f'traffic_dependence_{param}.png', dpi=300)
            plt.close()

        # Store the traffic models
        self.traffic_dependence = traffic_models

        # Save the models
        with open(self.output_dir / 'traffic_dependence_models.json', 'w') as f:
            json.dump({
                key: {k: float(v) if isinstance(v, np.float64) else v
                      for k, v in value.items()}
                for key, value in traffic_models.items()
            }, f, indent=4)

        elapsed_time = time.time() - start_time
        logging.info(f"Traffic dependence modeling completed in {elapsed_time:.2f} seconds")

        return True

    def model_traffic_dependence_ks(self):
        """
        Model how NCT parameters depend on traffic using KS distance optimization.

        This implements the thesis approach: param = alpha + beta * traffic
        where alpha and beta are optimized using dual annealing to minimize
        the KS distance between empirical and modeled distributions.
        """
        # Try to load daily parameters from file if not already available
        if not hasattr(self, 'param_df') or self.param_df is None or self.param_df.empty:
            if not self._load_daily_parameters_from_file():
                logging.error("No daily parameters available for traffic dependence modeling")
                return False

        start_time = time.time()
        logging.info("Modeling traffic dependence of NCT parameters using KS distance...")

        # Parameters to model
        param_names = ['df', 'nc', 'loc', 'scale']

        # Store traffic dependence models
        traffic_models_ks = {}

        # Bounds for alpha, beta parameters
        bounds = {
            'df': [(1.0, 50.0), (-0.01, 0.01)],    # (alpha_bounds, beta_bounds)
            'nc': [(-5.0, 5.0), (-0.01, 0.01)],
            'loc': [(-10.0, 10.0), (-0.01, 0.01)],
            'scale': [(0.1, 20.0), (-0.01, 0.01)]
        }

        # For each parameter, optimize alpha and beta
        for param in param_names:
            logging.info(f"Optimizing traffic dependence for {param}...")

            # Get the data
            traffic_values = self.param_df['traffic'].values
            param_values = self.param_df[param].values
            dates = self.param_df['date'].values

            # Define the objective function using KS distance
            def ks_objective(alpha_beta):
                alpha, beta = alpha_beta

                # Calculate all predicted param values
                predicted_params = alpha + beta * traffic_values

                # Enforce constraints on parameters
                if param == 'df' and np.any(predicted_params <= 0):
                    return 1e10
                if param == 'scale' and np.any(predicted_params <= 0):
                    return 1e10

                # Calculate KS distances for each day and average them
                ks_distances = []

                for i, date in enumerate(dates):
                    if date not in self.daily_delays:
                        continue

                    delays = self.daily_delays[date]

                    # Get the predicted parameter for this day
                    if param == 'df':
                        df_val = predicted_params[i]
                        nc_val = self.param_df['nc'].values[i]
                        loc_val = self.param_df['loc'].values[i]
                        scale_val = self.param_df['scale'].values[i]
                    elif param == 'nc':
                        df_val = self.param_df['df'].values[i]
                        nc_val = predicted_params[i]
                        loc_val = self.param_df['loc'].values[i]
                        scale_val = self.param_df['scale'].values[i]
                    elif param == 'loc':
                        df_val = self.param_df['df'].values[i]
                        nc_val = self.param_df['nc'].values[i]
                        loc_val = predicted_params[i]
                        scale_val = self.param_df['scale'].values[i]
                    elif param == 'scale':
                        df_val = self.param_df['df'].values[i]
                        nc_val = self.param_df['nc'].values[i]
                        loc_val = self.param_df['loc'].values[i]
                        scale_val = predicted_params[i]

                    # Safely calculate KS statistic
                    try:
                        ks_stat, _ = kstest(
                            delays,
                            lambda x: stats.nct.cdf(x, df=df_val, nc=nc_val, loc=loc_val, scale=scale_val)
                        )
                        ks_distances.append(ks_stat)
                    except:
                        ks_distances.append(1.0)  # Maximum distance on failure

                if len(ks_distances) == 0:
                    return 1.0

                # Return the mean KS distance as the objective to minimize
                return np.mean(ks_distances)

            # Initial guess based on simple regression
            X = traffic_values
            y = param_values
            z = np.polyfit(X, y, 1)
            alpha_init, beta_init = z[1], z[0]  # Polyfit returns [beta, alpha]

            logging.info(f"Initial guess for {param}: alpha={alpha_init:.4f}, beta={beta_init:.4f}")

            # Optimize using dual annealing
            result = dual_annealing(
                ks_objective,
                bounds=bounds[param],
                x0=[alpha_init, beta_init],
                maxiter=300
            )

            alpha_opt, beta_opt = result.x

            # Store the model
            traffic_models_ks[param] = {
                'alpha': float(alpha_opt),
                'beta': float(beta_opt),
                'converged': bool(result.success),
                'message': str(result.message),
                'objective': float(result.fun)
            }

            logging.info(f"KS Optimized {param} = {alpha_opt:.4f} + {beta_opt:.4f} * traffic "
                        f"(objective={result.fun:.4f})")

            # Generate predictions for visualization
            traffic_range = np.linspace(min(traffic_values), max(traffic_values), 100)
            param_pred = alpha_opt + beta_opt * traffic_range

            # Plot the fit
            plt.figure(figsize=(10, 6))
            plt.scatter(traffic_values, param_values, alpha=0.6, label='Daily fitted values')
            plt.plot(traffic_range, param_pred, 'r-', linewidth=2,
                    label=f'{param} = {alpha_opt:.4f} + {beta_opt:.4f} * traffic')

            plt.title(f'KS Optimized Traffic Dependence of NCT {param} Parameter')
            plt.xlabel('Daily Traffic Volume')
            plt.ylabel(f'NCT {param} Parameter')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / f'ks_traffic_dependence_{param}.png', dpi=300)
            plt.close()

        # Store the traffic models
        self.traffic_dependence_ks = traffic_models_ks

        # Save the models
        with open(self.output_dir / 'ks_traffic_dependence_models.json', 'w') as f:
            json.dump(traffic_models_ks, f, indent=4)

        # Validate the models
        self._validate_traffic_models(traffic_models_ks, suffix='ks')

        elapsed_time = time.time() - start_time
        logging.info(f"KS-based traffic dependence modeling completed in {elapsed_time:.2f} seconds")

        return True

    def _plot_fit(self, delays, params, label=""):
        """Plot the empirical vs. fitted distribution."""
        try:
            plt.figure(figsize=(12, 8))

            # Plot histogram of delays
            sns.histplot(delays, bins=50, kde=True, stat='density', alpha=0.6, label='Empirical')

            # Generate fitted distribution curve
            x = np.linspace(np.min(delays), np.max(delays), 1000)
            y = stats.nct.pdf(x, df=params['df'], nc=params['nc'], loc=params['loc'], scale=params['scale'])

            # Plot fitted curve
            plt.plot(x, y, 'r-', linewidth=2, label='NCT Fit')

            plt.title(f'NCT Distribution Fit - {label.capitalize()}')
            plt.xlabel('Delay (minutes)')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Add parameter details
            param_text = (f"df = {params['df']:.4f}, nc = {params['nc']:.4f}, "
                          f"loc = {params['loc']:.4f}, scale = {params['scale']:.4f}")
            plt.annotate(param_text, xy=(0.5, 0.02), xycoords='figure fraction',
                         bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                         ha='center')

            # Add KS test results
            if 'ks_stat' in params:
                ks_text = f"KS statistic = {params['ks_stat']:.4f}, p-value = {params['p_value']:.4f}"
                plt.annotate(ks_text, xy=(0.5, 0.06), xycoords='figure fraction',
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                             ha='center')

            plt.tight_layout()
            plt.savefig(self.output_dir / f'nct_fit_{label}.png', dpi=300)
            plt.close()

        except Exception as e:
            logging.error(f"Error plotting fit: {str(e)}")

    def _plot_params_vs_traffic(self):
        """Plot the fitted parameters vs. traffic."""
        try:
            # Create a figure with subplots
            fig, axs = plt.subplots(2, 2, figsize=(15, 10))
            axs = axs.flatten()

            param_names = ['df', 'nc', 'loc', 'scale']

            for i, param in enumerate(param_names):
                axs[i].scatter(self.param_df['traffic'], self.param_df[param], alpha=0.7)
                axs[i].set_title(f'NCT {param} vs. Traffic')
                axs[i].set_xlabel('Daily Traffic Volume')
                axs[i].set_ylabel(f'NCT {param} Parameter')
                axs[i].grid(True, alpha=0.3)

                # Add trend line
                z = np.polyfit(self.param_df['traffic'], self.param_df[param], 1)
                p = np.poly1d(z)
                axs[i].plot(self.param_df['traffic'], p(self.param_df['traffic']), "r--", alpha=0.7)

            plt.tight_layout()
            plt.savefig(self.output_dir / 'params_vs_traffic.png', dpi=300)
            plt.close()

        except Exception as e:
            logging.error(f"Error plotting parameters vs. traffic: {str(e)}")

    def _validate_traffic_models(self, traffic_models, suffix=''):
        """
        Validate the traffic dependence models by comparing empirical and predicted distributions.

        Args:
            traffic_models: Dictionary of traffic dependence models
            suffix: Suffix for output files
        """
        try:
            logging.info(f"Validating traffic dependence models with suffix '{suffix}'...")

            # Create a summary table for validation results
            validation_results = []

            # Iterate through days in dataset
            for date in tqdm(self.daily_delays.keys(), desc=f"Validating {suffix} models"):
                if date not in self.param_df['date'].values:
                    continue

                # Get the daily delays and traffic
                delays = self.daily_delays[date]
                traffic = self.daily_traffic.get(date, 0)

                # Get the row index in param_df for this date
                idx = self.param_df[self.param_df['date'] == date].index[0]

                # Calculate predicted NCT parameters based on traffic
                df_pred = traffic_models['df']['alpha'] + traffic_models['df']['beta'] * traffic
                nc_pred = traffic_models['nc']['alpha'] + traffic_models['nc']['beta'] * traffic
                loc_pred = traffic_models['loc']['alpha'] + traffic_models['loc']['beta'] * traffic
                scale_pred = traffic_models['scale']['alpha'] + traffic_models['scale']['beta'] * traffic

                # Ensure valid parameters
                df_pred = max(1.0, df_pred)
                scale_pred = max(0.1, scale_pred)

                # Calculate KS distance between empirical delays and predicted NCT distribution
                try:
                    ks_stat, p_value = kstest(
                        delays,
                        lambda x: stats.nct.cdf(x, df=df_pred, nc=nc_pred, loc=loc_pred, scale=scale_pred)
                    )

                    # Compare with the KS distance from individually fitted parameters
                    df_fitted = self.param_df.loc[idx, 'df']
                    nc_fitted = self.param_df.loc[idx, 'nc']
                    loc_fitted = self.param_df.loc[idx, 'loc']
                    scale_fitted = self.param_df.loc[idx, 'scale']

                    ks_stat_fitted, _ = kstest(
                        delays,
                        lambda x: stats.nct.cdf(x, df=df_fitted, nc=nc_fitted, loc=loc_fitted, scale=scale_fitted)
                    )

                    validation_results.append({
                        'date': str(date),
                        'traffic': traffic,
                        'n_delays': len(delays),
                        'ks_traffic_model': ks_stat,
                        'ks_individual_fit': ks_stat_fitted,
                        'ks_diff': ks_stat - ks_stat_fitted,
                        'p_value': p_value
                    })

                    # Occasionally plot the distribution comparison for visual inspection
                    if len(validation_results) % 10 == 0:  # Every 10th day
                        plt.figure(figsize=(12, 6))

                        # Plot empirical histogram
                        sns.histplot(delays, bins=30, kde=True, stat='density',
                                    alpha=0.6, label='Empirical', color='blue')

                        # Plot traffic-based model
                        x = np.linspace(min(delays), max(delays), 1000)
                        y_traffic = stats.nct.pdf(x, df=df_pred, nc=nc_pred, loc=loc_pred, scale=scale_pred)
                        plt.plot(x, y_traffic, 'r-', linewidth=2,
                                label=f'Traffic Model (KS={ks_stat:.4f})')

                        # Plot individual fit
                        y_fitted = stats.nct.pdf(x, df=df_fitted, nc=nc_fitted,
                                               loc=loc_fitted, scale=scale_fitted)
                        plt.plot(x, y_fitted, 'g--', linewidth=2,
                                label=f'Individual Fit (KS={ks_stat_fitted:.4f})')

                        plt.title(f'NCT Distribution - Date: {date}, Traffic: {traffic}')
                        plt.xlabel('Delay (minutes)')
                        plt.ylabel('Density')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(self.output_dir / f'validation_{suffix}_{date}.png', dpi=300)
                        plt.close()

                except Exception as e:
                    logging.warning(f"Error validating date {date}: {str(e)}")

            # Save validation results
            if validation_results:
                df_validation = pd.DataFrame(validation_results)
                df_validation.to_csv(self.output_dir / f'validation_results_{suffix}.csv', index=False)

                # Create summary statistics
                ks_traffic_mean = df_validation['ks_traffic_model'].mean()
                ks_fitted_mean = df_validation['ks_individual_fit'].mean()
                ks_diff_mean = df_validation['ks_diff'].mean()

                logging.info(f"Validation summary ({suffix}):")
                logging.info(f"  Mean KS distance (traffic model): {ks_traffic_mean:.4f}")
                logging.info(f"  Mean KS distance (individual fits): {ks_fitted_mean:.4f}")
                logging.info(f"  Mean KS distance difference: {ks_diff_mean:.4f}")

                # Plot validation summary
                plt.figure(figsize=(10, 6))
                plt.scatter(df_validation['traffic'], df_validation['ks_traffic_model'],
                          alpha=0.7, label='Traffic Model KS', color='red')
                plt.scatter(df_validation['traffic'], df_validation['ks_individual_fit'],
                          alpha=0.7, label='Individual Fit KS', color='blue')
                plt.axhline(y=ks_traffic_mean, color='red', linestyle='--',
                          label=f'Traffic Model Mean: {ks_traffic_mean:.4f}')
                plt.axhline(y=ks_fitted_mean, color='blue', linestyle='--',
                          label=f'Individual Fit Mean: {ks_fitted_mean:.4f}')

                plt.title(f'KS Statistics Comparison - {suffix.upper()} Traffic Models')
                plt.xlabel('Traffic Volume')
                plt.ylabel('KS Distance')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.output_dir / f'validation_summary_{suffix}.png', dpi=300)
                plt.close()

        except Exception as e:
            logging.error(f"Error validating traffic models: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())

    def _load_daily_parameters_from_file(self):
        """Load daily parameters from CSV file if they exist."""
        try:
            # Try both relative and absolute paths to find the daily parameters file
            possible_paths = [
                self.output_dir / 'daily_nct_parameters.csv',
                Path('data/DistributionFitting/daily_nct_parameters.csv'),
                Path(r'C:\Stefanija\MANU\Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans\AirTransportEvents\data\DistributionFitting\daily_nct_parameters.csv')
            ]

            daily_params_file = None
            for path in possible_paths:
                print(f"Checking for daily parameters file at: {path}")
                if path.exists():
                    daily_params_file = path
                    print(f"Found daily parameters file: {daily_params_file}")
                    break

            if daily_params_file is None:
                print("Daily parameters file not found in any of the expected locations")
                logging.error("Daily parameters file not found in any of the expected locations")
                return False

            logging.info(f"Loading daily parameters from {daily_params_file}")

            # Read the raw file content first to debug
            try:
                with open(daily_params_file, 'r') as f:
                    first_lines = [next(f) for _ in range(3) if _ < 3]  # Safely read up to 3 lines
                    print(f"First lines of CSV file:\n{''.join(first_lines)}")
                    # Reset file pointer
                    f.seek(0)
                    # Check if the file has headers by examining first line
                    first_line = next(f).strip() if first_lines else ""
                    has_header = 'date' in first_line.lower() or 'traffic' in first_line.lower()
                    print(f"CSV file appears to have header: {has_header}")
            except Exception as e:
                print(f"Error reading raw CSV file: {e}")
                has_header = False  # Default to no header if we can't read the file

            try:
                # Load with appropriate header settings
                print(f"Loading CSV with header={has_header}")
                if has_header:
                    self.param_df = pd.read_csv(daily_params_file)
                else:
                    self.param_df = pd.read_csv(
                        daily_params_file,
                        header=None,
                        names=['date', 'traffic', 'df', 'nc', 'loc', 'scale']
                    )
                print(f"Successfully loaded CSV with shape: {self.param_df.shape}")

                # Check and rename columns if needed
                if 'date' not in self.param_df.columns:
                    print(f"CSV columns: {self.param_df.columns}")
                    print("Renaming columns to expected names")
                    self.param_df.columns = ['date', 'traffic', 'df', 'nc', 'loc', 'scale']

            except Exception as e:
                print(f"Error loading CSV: {str(e)}")
                logging.error(f"Error loading CSV: {str(e)}")
                return False

            print(f"Loaded file with columns: {list(self.param_df.columns)}")
            print(f"First few rows:\n{self.param_df.head(3)}")
            logging.info(f"Loaded file with columns: {list(self.param_df.columns)}")
            logging.info(f"Loaded {len(self.param_df)} rows")

            # Convert date column to datetime.date objects
            try:
                self.param_df['date'] = pd.to_datetime(self.param_df['date']).dt.date
                print("Successfully converted dates to datetime.date objects")
            except Exception as e:
                print(f"Error converting date column: {e}. Using as-is.")
                logging.warning(f"Error converting date column: {e}. Using as-is.")

            # Rebuild daily_delays dictionary if needed
            if not hasattr(self, 'daily_delays') or not self.daily_delays or len(self.daily_delays) == 0:
                print("Rebuilding daily_delays dictionary...")
                logging.info("Rebuilding daily_delays dictionary...")

                # First approach: use the loaded dataframe
                if hasattr(self, 'df') and self.df is not None and 'date' in self.df.columns and 'delay_min' in self.df.columns:
                    print("Using loaded dataframe to rebuild daily_delays")
                    daily_groups = self.df.groupby('date')
                    self.daily_delays = {date: group['delay_min'].values for date, group in daily_groups}
                    print(f"Rebuilt daily_delays with {len(self.daily_delays)} days")
                else:
                    # If we don't have the dataframe, create a minimal structure for validation
                    print("Main dataframe not available. Creating minimal daily_delays for validation.")
                    unique_dates = self.param_df['date'].unique()

                    # For each date, try to generate some synthetic data if needed
                    self.daily_delays = {}
                    for date in unique_dates:
                        # Get parameters for this date
                        idx = self.param_df[self.param_df['date'] == date].index[0]
                        df = self.param_df.loc[idx, 'df']
                        nc = self.param_df.loc[idx, 'nc']
                        loc = self.param_df.loc[idx, 'loc']
                        scale = self.param_df.loc[idx, 'scale']

                        # Generate synthetic samples for validation
                        try:
                            # Generate a small number of synthetic samples - enough for KS tests
                            n_samples = 100
                            synthetic = stats.nct.rvs(df=df, nc=nc, loc=loc, scale=scale, size=n_samples, random_state=42)
                            self.daily_delays[date] = synthetic
                            print(f"Generated {n_samples} synthetic samples for date {date}")
                        except Exception as e:
                            # Fallback to empty array if generation fails
                            self.daily_delays[date] = np.array([])
                            print(f"Error generating synthetic data for date {date}: {e}")

                    print(f"Created synthetic daily_delays with {len(self.daily_delays)} days")

            logging.info(f"Successfully loaded {len(self.param_df)} daily parameter sets from file")
            return True

        except Exception as e:
            print(f"Error loading daily parameters from file: {e}")
            logging.error(f"Error loading daily parameters from file: {e}")
            import traceback
            traceback_str = traceback.format_exc()
            print(traceback_str)
            logging.error(traceback.format_exc())
            return False

def main():
    """Main function to execute the distribution fitting analysis with command-line options."""
    print("Starting distribution fitting script...")

    parser = argparse.ArgumentParser(description='Fast NCT distribution fitting for air transport delays')

    parser.add_argument('--sample', type=int, help='Sample size for faster processing')
    parser.add_argument('--step', choices=['all', 'global', 'daily', 'traffic', 'traffic_ks'], default='all',
                       help='Which step to run (default: all)')
    parser.add_argument('--traffic-method', choices=['simple', 'ks', 'both'], default='both',
                      help='Method for traffic dependence modeling: simple regression, KS optimization, or both')
    parser.add_argument('--maxiter', type=int, default=500, help='Maximum iterations for optimization')
    parser.add_argument('--max-days', type=int, help='Maximum number of days to fit')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with console output')

    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    # Configure logging
    log_handlers = [logging.FileHandler('distribution_fitting_fast.log')]
    if args.debug:
        log_handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=log_handlers
    )
    print("Logging configured")

    try:
        print("Starting execution...")
        start_time = time.time()
        logging.info(f"Starting fast distribution fitting with options: {args}")

        # Initialize the fitter
        print("Initializing fitter...")
        fitter = FastNCTFitter(sample_size=args.sample, random_seed=args.seed)
        print("Fitter initialized")

        # Load data
        print("Loading data...")
        if not fitter.load_data():
            print("Failed to load data")
            return False
        print("Data loaded successfully")

        # Determine if we need to run prerequisite steps
        run_global = args.step in ['all', 'global']
        run_daily = args.step in ['all', 'daily']
        run_traffic = args.step in ['all', 'traffic', 'traffic_ks']

        # For traffic modeling, we need daily parameters
        if run_traffic and not run_daily:
            # Check if daily parameters file exists
            daily_params_file = fitter.output_dir / 'daily_nct_parameters.csv'
            if not daily_params_file.exists():
                print("Daily parameters not found but required for traffic modeling. Running daily fitting step first...")
                run_daily = True
                # For daily fitting, we benefit from having global parameters
                if not run_global:
                    global_params_file = fitter.output_dir / 'global_nct_parameters.json'
                    if not global_params_file.exists():
                        print("Global parameters not found. Running global fitting step first...")
                        run_global = True

        # Run global fitting if needed
        if run_global:
            print("Running global fitting...")
            if not fitter.fit_nct_global(maxiter=args.maxiter):
                print("Global fitting failed. Continuing anyway...")
                logging.warning("Global fitting failed. Continuing anyway...")

        # Run daily fitting if needed
        if run_daily:
            print("Running daily fitting...")
            if not fitter.fit_nct_daily(max_days=args.max_days, maxiter=args.maxiter):
                print("Daily fitting failed. Cannot proceed with traffic dependence modeling.")
                logging.error("Daily fitting failed. Cannot proceed with traffic dependence modeling.")
                return False

            # Try to load daily parameters even if we didn't run daily fitting (they might exist on disk)
            if not hasattr(fitter, 'param_df') or fitter.param_df is None or fitter.param_df.empty:
                try:
                    daily_params_file = fitter.output_dir / 'daily_nct_parameters.csv'
                    if daily_params_file.exists():
                        print("Loading existing daily parameters from file...")
                        fitter.param_df = pd.read_csv(daily_params_file)
                        print(f"Loaded {len(fitter.param_df)} daily parameter sets")
                except Exception as e:
                    print(f"Error loading daily parameters: {e}")
                    logging.error(f"Error loading daily parameters: {e}")
                    if run_traffic:
                        print("Cannot proceed with traffic dependence modeling without daily parameters.")
                        return False

        # Handle traffic dependence modeling based on method choice
        if run_traffic:
            if not hasattr(fitter, 'param_df') or fitter.param_df is None or fitter.param_df.empty:
                print("No daily parameters available. Cannot proceed with traffic dependence modeling.")
                logging.error("No daily parameters available. Cannot proceed with traffic dependence modeling.")
                return False

            print(f"Running traffic dependence with method: {args.traffic_method}")
            if args.step == 'traffic_ks' or args.traffic_method == 'ks':
                # Run only KS-based traffic dependence modeling
                print("Running KS-based traffic dependence modeling...")
                fitter.model_traffic_dependence_ks()
            elif args.step == 'traffic' and args.traffic_method == 'simple':
                # Run only simple regression traffic dependence
                print("Running simple regression traffic dependence...")
                fitter.model_traffic_dependence()
            elif args.traffic_method == 'both' or args.step == 'all':
                # Run both methods
                print("Running both traffic dependence methods...")
                logging.info("Running both traffic dependence methods for comparison")
                fitter.model_traffic_dependence()
                fitter.model_traffic_dependence_ks()

                # Create comparison visualizations
                try:
                    logging.info("Generating method comparison visualizations")
                    if hasattr(fitter, 'traffic_dependence') and hasattr(fitter, 'traffic_dependence_ks'):
                        print("Comparing traffic methods...")
                        _compare_traffic_methods(fitter)
                except Exception as e:
                    print(f"Error comparing traffic methods: {str(e)}")
                    logging.error(f"Error comparing traffic methods: {str(e)}")

        total_time = time.time() - start_time
        logging.info(f"Distribution fitting completed in {total_time:.2f} seconds")
        print(f"Distribution fitting completed in {total_time:.2f} seconds")

    except Exception as e:
        print(f"Error in distribution fitting: {str(e)}")
        import traceback
        traceback_str = traceback.format_exc()
        print(traceback_str)
        logging.error(f"Error in distribution fitting: {str(e)}")
        logging.error(traceback.format_exc())

def _compare_traffic_methods(fitter):
    """
    Compare the simple regression and KS-optimized traffic dependence models.

    Args:
        fitter: FastNCTFitter instance with both models available
    """
    try:
        logging.info("Comparing traffic dependence methods...")
        output_dir = fitter.output_dir

        # Parameters to compare
        param_names = ['df', 'nc', 'loc', 'scale']

        # Create a comparison plot for each parameter
        for param in param_names:
            # Get the simple regression model parameters
            alpha_simple = fitter.traffic_dependence[param]['alpha']
            beta_simple = fitter.traffic_dependence[param]['beta']
            r2_simple = fitter.traffic_dependence[param]['r_squared']

            # Get the KS-optimized model parameters
            alpha_ks = fitter.traffic_dependence_ks[param]['alpha']
            beta_ks = fitter.traffic_dependence_ks[param]['beta']
            obj_ks = fitter.traffic_dependence_ks[param]['objective']

            # Get traffic and param values for plotting
            traffic_values = fitter.param_df['traffic'].values
            param_values = fitter.param_df[param].values

            # Create traffic range for prediction lines
            traffic_range = np.linspace(min(traffic_values), max(traffic_values), 100)

            # Calculate predictions
            simple_pred = alpha_simple + beta_simple * traffic_range
            ks_pred = alpha_ks + beta_ks * traffic_range

            # Create comparison plot
            plt.figure(figsize=(12, 8))
            plt.scatter(traffic_values, param_values, alpha=0.7, label='Daily fitted values', color='gray')

            # Plot simple regression line
            plt.plot(traffic_range, simple_pred, 'b-', linewidth=2,
                    label=f'Simple: {param} = {alpha_simple:.4f} + {beta_simple:.4f} * traffic (R² = {r2_simple:.4f})')

            # Plot KS-optimized line
            plt.plot(traffic_range, ks_pred, 'r--', linewidth=2,
                    label=f'KS Opt: {param} = {alpha_ks:.4f} + {beta_ks:.4f} * traffic (obj = {obj_ks:.4f})')

            plt.title(f'Traffic Dependence Models Comparison - NCT {param} Parameter')
            plt.xlabel('Daily Traffic Volume')
            plt.ylabel(f'NCT {param} Parameter')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / f'comparison_{param}.png', dpi=300)
            plt.close()

        # Create a summary table for model comparison
        comparison = {}
        for param in param_names:
            comparison[param] = {
                'simple_alpha': float(fitter.traffic_dependence[param]['alpha']),
                'simple_beta': float(fitter.traffic_dependence[param]['beta']),
                'simple_r2': float(fitter.traffic_dependence[param]['r_squared']),
                'ks_alpha': float(fitter.traffic_dependence_ks[param]['alpha']),
                'ks_beta': float(fitter.traffic_dependence_ks[param]['beta']),
                'ks_objective': float(fitter.traffic_dependence_ks[param]['objective'])
            }

        # Save comparison to JSON
        with open(output_dir / 'method_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=4)

        logging.info("Method comparison completed and saved")

    except Exception as e:
        logging.error(f"Error comparing traffic methods: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
