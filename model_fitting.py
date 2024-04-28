import numpy as np
import pandas as pd
import scipy
import black_scholes_formula
from scipy.interpolate import UnivariateSpline

def rmse(market, model) -> np.float64:
    return np.sqrt((np.power(market-model, 2)).mean())


class PDF(scipy.stats.rv_continuous):
    def _pdf(self):
        # am thinking could pass an interpolator of RNDs for this...
        pass




class ModelFit(object):

    def __init__(self,  obj_func, intial_guess, bounds, param_names, vol_func):
        """
        :param obj_func: objective function to be passed to Scipy Minimise
        :param intial_guess: initial guess for params of obj function
        :param bounds: bounds for these parameters, an argument of scipy minimise
        :param param_names: just for printing
        """

        self.initial_guess = intial_guess
        self.bounds = bounds
        self.obj_func = obj_func
        self.param_names = param_names
        self.vol_func = vol_func

    def perform_minimisation(self, obj_func_args) -> scipy.optimize._optimize.OptimizeResult:
        result = scipy.optimize.minimize(self.obj_func,
                                         x0=self.initial_guess.copy(),
                                         args=obj_func_args,
                                         bounds=self.bounds)
        self.result = result

        return result

    def fit_cubic_spline(self, calls):
        # kind='cubic', fill_value="extrapolate"
        y = calls.ImpliedVolatility.values
        # vol_function = scipy.interpolate.interp1d(x=calls.Strike.values, y=calls.ImpliedVolatility.values,
        #                                          kind='cubic', fill_value=(y[0], y[-1]), bounds_error=False)
        # #(y[0], y[-1])
        vol_function = UnivariateSpline(x=calls.Strike.values,
                                        y=calls.ImpliedVolatility.values,
                                        k=3,
                                        s=0.05,
                                        ext=3) # ext=3 returns boundary values!

        self.vol_func = vol_function

        return vol_function



    def show_result(self, r=None):
        if r is None:
            r = self.result

        return dict(zip(self.param_names, r.x))

    def calc_call_price_errors(self, call_data, fwd, tenor, rate, vol_func_arg):
        """
        :param call_data: call options for specific date and expiry. e.g. all_1M_wk.loc[date, "C"]
        :return:
        """
        strike_range = call_data.Strike.values
        mkt_prices = call_data.mid.values
        call_prices_interp = np.zeros(len(strike_range))

        for i, strike in enumerate(strike_range):
            vol_func_arg["K"] = strike
            call_price_interp = black_scholes_formula.black_scholes_call(S=fwd, K=strike,
                                                                         t=tenor,
                                                                         sigma=self.vol_func(**vol_func_arg),
                                                                         r=rate)
            call_prices_interp[i] = call_price_interp
        error_pct = (call_prices_interp-mkt_prices)/mkt_prices

        return error_pct

    def check_butterflies(self, calls):
        data = []
        for (_, left), (_, centre), (_, right) in zip(calls.iterrows(), calls.iloc[1:].iterrows(),
                                                      calls.iloc[2:].iterrows()):
            if centre.Strike - left.Strike != right.Strike - centre.Strike:
                continue
            butterfly_price = left.mid - 2 * centre.mid + right.mid
            max_profit = centre.Strike - left.Strike
            data.append([centre.Strike, butterfly_price, max_profit])

        bflys = pd.DataFrame(data, columns=["strike", "price", "max_profit"])
        bflys["prob"] = bflys.price / bflys.max_profit
        return bflys

    def get_model_calls(self, krange,fwd, tenor, rate, vol_func_arg):
        call_prices_interp = np.zeros(len(krange))
        for i, strike in enumerate(krange):
            vol_func_arg["K"] = strike
            call_price_interp = black_scholes_formula.black_scholes_call(S=fwd, K=strike,
                                                                         t=tenor,
                                                                         sigma=self.vol_func(**vol_func_arg),
                                                                         r=rate)
            call_prices_interp[i] = call_price_interp
        return call_prices_interp

    def construct_pdf(self, min_strike, max_strike, fwd, tenor, rate, vol_func_arg):
        k_new = np.arange(start=min_strike, stop=max_strike, step=1)
        # unfortuntaely for now will have to loop: # should store this as attribute for later use
        call_prices_interp = np.zeros(len(k_new))  # Allocate all memory first
        for i, strike in enumerate(k_new):
            vol_func_arg["K"] = strike
            call_price_interp = black_scholes_formula.black_scholes_call(S=fwd, K=strike,
                                                                         t=tenor,
                                                                         sigma=self.vol_func(**vol_func_arg),
                                                                         r=rate)

            call_prices_interp[i] = call_price_interp
        first_deriv = np.gradient(call_prices_interp, k_new, edge_order=0)
        second_deriv = np.gradient(first_deriv, k_new, edge_order=0)
        densities = np.exp(-rate*tenor)*second_deriv
        return densities

    def construct_pdf_cubic(self, min_strike, max_strike, fwd, tenor, rate):
        k_new = np.arange(start=min_strike, stop=max_strike, step=1)
        call_prices_interp = black_scholes_formula.black_scholes_call(S=fwd,
                                                                     K=k_new,
                                                                     sigma=self.vol_func(k_new),
                                                                     t=tenor, r=rate)

        first_deriv = np.gradient(call_prices_interp, k_new, edge_order=0)
        second_deriv = np.gradient(first_deriv, k_new, edge_order=0)
        densities = np.exp(-rate*tenor)*second_deriv
        return densities

    def calc_call_price_errors_cubic(self, call_data, fwd, tenor, rate):
        strike_range = call_data.Strike.values
        mkt_prices = call_data.mid.values
        call_prices_interp = black_scholes_formula.black_scholes_call(S=fwd,
                                                                      K=strike_range,
                                                                      sigma=self.vol_func(strike_range),
                                                                      t=tenor, r=rate)

        error_pct = (call_prices_interp - mkt_prices) / mkt_prices
        return error_pct

    def get_model_calls_cubic(self, krange, fwd, tenor, rate):
        call_prices_interp = black_scholes_formula.black_scholes_call(S=fwd,
                                                                      K=krange,
                                                                      sigma=self.vol_func(krange),
                                                                      t=tenor,
                                                                      r=rate)
        return call_prices_interp
