"""
Distribution Fitting and Modeling for Air Transport Delays

This script fits noncentral Student's t-distributions (NCT) to air transport delay data
and models how distribution parameters depend on traffic volume, following the approach
described in the master's thesis.

Key features:
- Fits NCT distribution to daily delay histograms or the whole dataset
- Models traffic-dependent parameters (df, nc, loc, scale) with linear traffic dependence
- Uses dual annealing for global optimization of parameter dependencies
- Compares empirical and fitted distributions using KS distance
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
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('distribution_fitting.log'),
        logging.StreamHandler()
    ]
)

class NCTDistributionFitter:
    """Class for fitting noncentral Student's t-distributions to delay data and modeling traffic dependence."""

    def __init__(self, data_dir="data/ProcessedData", output_dir="data/DistributionFitting"):
        """Initialize the NCTDistributionFitter."""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Store fitted parameters
        self.params = {}
        self.traffic_dependence = {}
        self.daily_data = {}

    def load_data(self):
        """Load the delay data and traffic volumes."""
        try:
            # Load cleaned delay data from parquet file
            df_path = self.data_dir / 'cleaned_delays.parquet'
            logging.info(f"Loading delay data from {df_path}")

            self.df = pd.read_parquet(df_path)

            # First, let's print the column names to debug
            column_names = self.df.columns.tolist()
            logging.info(f"Available columns: {column_names}")

            # Extract delay values and traffic data
            # Using delay_min column instead of delay
            self.delays = self.df['delay_min'].values

            # Prepare daily data
            # Using sched_dt_utc for date extraction
            self.df['date'] = pd.to_datetime(self.df['sched_dt_utc']).dt.date
            self.daily_traffic = self.df.groupby('date').size()

            daily_groups = self.df.groupby('date')
            self.daily_delays = {date: group['delay_min'].values for date, group in daily_groups}

            logging.info(f"Loaded {len(self.delays)} delay records across {len(self.daily_delays)} days")
            return True

        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())  # Print full traceback for better debugging
            return False

    def fit_nct_global(self):
        """Fit the noncentral Student's t-distribution to the entire dataset."""
        try:
            logging.info("Fitting global NCT distribution...")

            # Initial parameter guesses [df, nc, loc, scale]
            # df: degrees of freedom, nc: noncentrality parameter
            # loc: location parameter, scale: scale parameter
            initial_params = [5.0, 0.0, 0.0, 1.0]

            # Fit NCT using MLE or method of moments
            # Starting with bounded optimization
            bounds = [(1.0, 100.0),  # df: degrees of freedom > 1
                      (-10.0, 10.0),  # nc: noncentrality parameter
                      (-20.0, 20.0),  # loc: location parameter
                      (0.1, 50.0)]    # scale: scale parameter > 0

            # Define the negative log-likelihood function for NCT
            def neg_log_likelihood(params):
                df, nc, loc, scale = params
                # Avoid invalid parameters
                if df <= 0 or scale <= 0:
                    return 1e10
                try:
                    # Calculate log PDF for each data point
                    log_pdf = np.sum(stats.nct.logpdf(self.delays, df=df, nc=nc, loc=loc, scale=scale))
                    # Return negative log-likelihood
                    return -log_pdf
                except:
                    # Return a large value if calculation fails
                    return 1e10

            # Optimize using a global optimizer (dual annealing)
            result = dual_annealing(neg_log_likelihood, bounds=bounds, maxiter=1000)

            if result.success:
                df_opt, nc_opt, loc_opt, scale_opt = result.x
                self.params['global'] = {
                    'df': df_opt,
                    'nc': nc_opt,
                    'loc': loc_opt,
                    'scale': scale_opt,
                    'converged': True,
                    'message': result.message
                }

                # Calculate KS statistic for goodness of fit
                ks_stat, p_value = kstest(
                    self.delays,
                    lambda x: stats.nct.cdf(x, df=df_opt, nc=nc_opt, loc=loc_opt, scale=scale_opt)
                )

                self.params['global']['ks_stat'] = ks_stat
                self.params['global']['p_value'] = p_value

                logging.info(f"Global NCT fit: df={df_opt:.4f}, nc={nc_opt:.4f}, loc={loc_opt:.4f}, scale={scale_opt:.4f}")
                logging.info(f"KS test: statistic={ks_stat:.4f}, p-value={p_value:.4f}")

                # Visualize the fit
                self._plot_fit(self.delays, self.params['global'], "global")

                return True
            else:
                logging.error(f"Global fitting did not converge: {result.message}")
                return False

        except Exception as e:
            logging.error(f"Error fitting global NCT: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())  # Print full traceback for better debugging
            return False

    def fit_nct_daily(self):
        """Fit the noncentral Student's t-distribution to daily delay data."""
        logging.info("Fitting daily NCT distributions...")

        # Initial parameter guesses and bounds
        initial_params = [5.0, 0.0, 0.0, 1.0]
        bounds = [(1.0, 100.0),    # df: degrees of freedom > 1
                 (-10.0, 10.0),    # nc: noncentrality parameter
                 (-20.0, 20.0),    # loc: location parameter
                 (0.1, 50.0)]      # scale: scale parameter > 0

        # Store daily fitted parameters
        daily_params = {}

        # Store traffic and parameter values for regression
        traffic_values = []
        df_values = []
        nc_values = []
        loc_values = []
        scale_values = []

        # Fit each day with sufficient data
        for date, delays in tqdm(self.daily_delays.items(), desc="Fitting daily distributions"):
            # Skip days with too few data points
            if len(delays) < 30:
                continue

            # Define the negative log-likelihood function for this day's data
            def neg_log_likelihood(params):
                df, nc, loc, scale = params
                # Avoid invalid parameters
                if df <= 0 or scale <= 0:
                    return 1e10
                try:
                    # Calculate log PDF for each data point
                    log_pdf = np.sum(stats.nct.logpdf(delays, df=df, nc=nc, loc=loc, scale=scale))
                    # Return negative log-likelihood
                    return -log_pdf
                except:
                    # Return a large value if calculation fails
                    return 1e10

            try:
                # Optimize using dual annealing for global optimization
                result = dual_annealing(neg_log_likelihood, bounds=bounds, maxiter=500)

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
                        'message': result.message,
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

            except Exception as e:
                logging.warning(f"Error fitting NCT for {date}: {str(e)}")

        # Store the fitted parameters
        self.params['daily'] = daily_params

        # Create DataFrame for analysis
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

        logging.info(f"Successfully fitted NCT distributions for {len(daily_params)} days")
        return True

    def model_traffic_dependence(self):
        """Model how NCT parameters depend on traffic."""
        if not hasattr(self, 'param_df') or self.param_df.empty:
            logging.error("No daily parameters available for traffic dependence modeling")
            return False

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
        pd.DataFrame(traffic_models).to_json(self.output_dir / 'traffic_dependence_models.json', orient='columns')

        return True

    def optimize_traffic_model(self):
        """
        Optimize traffic dependence model using dual annealing and KS distance.
        This follows the thesis approach where param = alpha + beta * traffic.
        """
        if not hasattr(self, 'daily_delays') or not self.daily_delays:
            logging.error("No daily data available for traffic model optimization")
            return False

        logging.info("Optimizing traffic dependence model with KS distance...")

        # Combine all data for parameter estimation
        all_delays = np.concatenate(list(self.daily_delays.values()))
        all_traffic = np.concatenate([np.full(len(delays), self.daily_traffic.get(date, 0))
                                     for date, delays in self.daily_delays.items()])

        # Create data pairs for optimization
        delay_traffic_pairs = list(zip(all_delays, all_traffic))

        # Define the bounds for optimization
        # [df_alpha, df_beta, nc_alpha, nc_beta, loc_alpha, loc_beta, scale_alpha, scale_beta]
        bounds = [
            (1.0, 50.0),    # df_alpha: baseline degrees of freedom
            (-0.1, 0.1),    # df_beta: traffic effect on df
            (-5.0, 5.0),    # nc_alpha: baseline noncentrality
            (-0.1, 0.1),    # nc_beta: traffic effect on nc
            (-10.0, 10.0),  # loc_alpha: baseline location
            (-0.1, 0.1),    # loc_beta: traffic effect on loc
            (0.1, 20.0),    # scale_alpha: baseline scale
            (-0.1, 0.1)     # scale_beta: traffic effect on scale
        ]

        # Define the objective function using KS distance
        def ks_distance(params):
            df_alpha, df_beta, nc_alpha, nc_beta, loc_alpha, loc_beta, scale_alpha, scale_beta = params

            # Calculate traffic-dependent parameters for each data point
            dfs = df_alpha + df_beta * all_traffic
            ncs = nc_alpha + nc_beta * all_traffic
            locs = loc_alpha + loc_beta * all_traffic
            scales = scale_alpha + scale_beta * all_traffic

            # Ensure parameters are valid
            invalid = (dfs <= 0) | (scales <= 0)
            if np.any(invalid):
                return 1e10

            try:
                # Create samples from the model with traffic-dependent parameters
                model_samples = np.array([stats.nct.rvs(df=dfs[i], nc=ncs[i], loc=locs[i], scale=scales[i])
                                         for i in range(len(all_delays))])

                # Calculate KS distance between empirical and model distributions
                ks_stat, _ = ks_2samp(all_delays, model_samples)

                return ks_stat

            except Exception as e:
                logging.warning(f"Error in KS distance calculation: {str(e)}")
                return 1e10

        # Optimize using dual annealing
        try:
            result = dual_annealing(ks_distance, bounds=bounds, maxiter=1000)

            if result.success:
                df_alpha, df_beta, nc_alpha, nc_beta, loc_alpha, loc_beta, scale_alpha, scale_beta = result.x

                # Store optimized model
                traffic_model = {
                    'df': {'alpha': df_alpha, 'beta': df_beta},
                    'nc': {'alpha': nc_alpha, 'beta': nc_beta},
                    'loc': {'alpha': loc_alpha, 'beta': loc_beta},
                    'scale': {'alpha': scale_alpha, 'beta': scale_beta},
                    'ks_stat': result.fun,
                    'converged': True,
                    'message': str(result.message)
                }

                self.traffic_dependence_optimized = traffic_model

                logging.info("Optimized traffic dependence model:")
                logging.info(f"df = {df_alpha:.4f} + {df_beta:.4f} * traffic")
                logging.info(f"nc = {nc_alpha:.4f} + {nc_beta:.4f} * traffic")
                logging.info(f"loc = {loc_alpha:.4f} + {loc_beta:.4f} * traffic")
                logging.info(f"scale = {scale_alpha:.4f} + {scale_beta:.4f} * traffic")
                logging.info(f"KS distance: {result.fun:.4f}")

                # Save the model
                pd.DataFrame(traffic_model).to_json(self.output_dir / 'traffic_dependence_optimized.json', orient='columns')

                # Visualize the model
                self._plot_optimized_model(traffic_model)

                return True

            else:
                logging.error(f"Optimization did not converge: {result.message}")
                return False

        except Exception as e:
            logging.error(f"Error optimizing traffic model: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return False

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
            import traceback
            logging.error(traceback.format_exc())

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
            import traceback
            logging.error(traceback.format_exc())

    def _plot_optimized_model(self, model):
        """Plot the optimized traffic-dependent distribution model."""
        try:
            # Sample traffic values in observed range
            if hasattr(self, 'daily_traffic'):
                min_traffic = min(self.daily_traffic.values())
                max_traffic = max(self.daily_traffic.values())
                traffic_range = np.linspace(min_traffic, max_traffic, 5)
            else:
                traffic_range = np.array([100, 200, 300, 400, 500])

            plt.figure(figsize=(12, 8))

            # Generate distributions for different traffic levels
            x = np.linspace(-20, 20, 1000)  # Reasonable range for delays

            for traffic in traffic_range:
                # Calculate parameters for this traffic level
                df = model['df']['alpha'] + model['df']['beta'] * traffic
                nc = model['nc']['alpha'] + model['nc']['beta'] * traffic
                loc = model['loc']['alpha'] + model['loc']['beta'] * traffic
                scale = model['scale']['alpha'] + model['scale']['beta'] * traffic

                # Generate PDF
                y = stats.nct.pdf(x, df=df, nc=nc, loc=loc, scale=scale)

                # Plot the curve
                plt.plot(x, y, '-', linewidth=2, label=f'Traffic = {int(traffic)}')

            plt.title('Traffic-Dependent NCT Distribution Model')
            plt.xlabel('Delay (minutes)')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Add parameter equations
            text = (f"df = {model['df']['alpha']:.4f} + {model['df']['beta']:.4f} * traffic\n"
                    f"nc = {model['nc']['alpha']:.4f} + {model['nc']['beta']:.4f} * traffic\n"
                    f"loc = {model['loc']['alpha']:.4f} + {model['loc']['beta']:.4f} * traffic\n"
                    f"scale = {model['scale']['alpha']:.4f} + {model['scale']['beta']:.4f} * traffic")

            plt.annotate(text, xy=(0.02, 0.02), xycoords='figure fraction',
                         bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

            plt.tight_layout()
            plt.savefig(self.output_dir / 'traffic_dependent_model.png', dpi=300)
            plt.close()

            # Create QQ plots to check fit quality
            self._create_qq_plots(model)

        except Exception as e:
            logging.error(f"Error plotting optimized model: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())

    def _create_qq_plots(self, model):
        """Create QQ plots to check fit quality for different traffic levels."""
        try:
            # Group delays by traffic bins
            if not hasattr(self, 'daily_delays') or not hasattr(self, 'daily_traffic'):
                return

            all_delays = []
            all_traffic = []

            for date, delays in self.daily_delays.items():
                traffic = self.daily_traffic.get(date, 0)
                all_delays.extend(delays)
                all_traffic.extend([traffic] * len(delays))

            all_delays = np.array(all_delays)
            all_traffic = np.array(all_traffic)

            # Create traffic bins
            traffic_bins = np.linspace(min(all_traffic), max(all_traffic), 6)
            bin_labels = [f"{int(traffic_bins[i])} - {int(traffic_bins[i+1])}"
                          for i in range(len(traffic_bins)-1)]

            # Assign each delay to a traffic bin
            delay_bin_indices = np.digitize(all_traffic, traffic_bins[1:])

            # Create QQ plots for each traffic bin
            fig, axs = plt.subplots(2, 3, figsize=(18, 10))
            axs = axs.flatten()

            for i in range(len(bin_labels)):
                bin_delays = all_delays[delay_bin_indices == i]

                if len(bin_delays) < 30:
                    axs[i].text(0.5, 0.5, 'Insufficient data',
                                ha='center', va='center', transform=axs[i].transAxes)
                    continue

                # Calculate average traffic for this bin
                avg_traffic = np.mean(all_traffic[delay_bin_indices == i])

                # Calculate NCT parameters for this traffic level
                df = model['df']['alpha'] + model['df']['beta'] * avg_traffic
                nc = model['nc']['alpha'] + model['nc']['beta'] * avg_traffic
                loc = model['loc']['alpha'] + model['loc']['beta'] * avg_traffic
                scale = model['scale']['alpha'] + model['scale']['beta'] * avg_traffic

                # Generate theoretical quantiles
                theoretical_quantiles = stats.nct.ppf(
                    np.linspace(0.01, 0.99, 100),
                    df=df, nc=nc, loc=loc, scale=scale
                )

                # Generate empirical quantiles
                empirical_quantiles = np.quantile(bin_delays, np.linspace(0.01, 0.99, 100))

                # Plot QQ plot
                axs[i].scatter(theoretical_quantiles, empirical_quantiles, alpha=0.7)
                axs[i].plot([min(theoretical_quantiles), max(theoretical_quantiles)],
                            [min(theoretical_quantiles), max(theoretical_quantiles)],
                            'r--', alpha=0.7)

                axs[i].set_title(f'Traffic Bin: {bin_labels[i]} (n={len(bin_delays)})')
                axs[i].set_xlabel('Theoretical Quantiles')
                axs[i].set_ylabel('Empirical Quantiles')
                axs[i].grid(True, alpha=0.3)

                # Add parameter values
                param_text = (f"df = {df:.2f}, nc = {nc:.2f}\n"
                             f"loc = {loc:.2f}, scale = {scale:.2f}")
                axs[i].annotate(param_text, xy=(0.05, 0.95), xycoords='axes fraction',
                                va='top', bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

            # Remove unused subplots
            for i in range(len(bin_labels), len(axs)):
                fig.delaxes(axs[i])

            plt.tight_layout()
            plt.savefig(self.output_dir / 'traffic_model_qq_plots.png', dpi=300)
            plt.close()

        except Exception as e:
            logging.error(f"Error creating QQ plots: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())

    def run_analysis(self):
        """Run the full analysis workflow."""
        logging.info("Starting distribution fitting and modeling analysis...")

        # Step 1: Load data
        if not self.load_data():
            return False

        # Step 2: Fit NCT to the whole dataset
        self.fit_nct_global()

        # Step 3: Fit NCT to daily data
        self.fit_nct_daily()

        # Step 4: Model traffic dependence with simple approach
        self.model_traffic_dependence()

        # Step 5: Optimize traffic model with KS distance
        self.optimize_traffic_model()

        logging.info("Distribution fitting and modeling analysis completed!")
        return True

def main():
    """Main function to execute the distribution fitting and modeling analysis."""
    try:
        fitter = NCTDistributionFitter()
        fitter.run_analysis()

    except Exception as e:
        logging.error(f"Error in distribution fitting analysis: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
