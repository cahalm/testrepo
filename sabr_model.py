import numpy as np
from model_fitting import rmse
import scipy


def SABR(alpha: float, p: float, nu: float, F: float , K: float ,t: float, B: float =1.0) -> np.float64:
    """
    alpha:

    B: beta
    p: rho
    nu: v
    F: forward
    K: strike
    t: time
    """

    if F == K:
        fb = F** (1 - B)
        A = alpha / (fb)
        B = 1 + t * (((((1 - B) ** 2) * (alpha ** 2)) / (24 * (F ** (2 - 2 * B)))) + (p * B * alpha * nu) / (
                    4.0 * fb) + ((nu ** 2) * (2 - 3 * (p ** 2))) / (24))
        vol_black = A * B

    elif F != K:
        fkb = (F * K) ** ((1 - B) / 2.0)
        logFK = np.log(F / K)
        z = (nu / alpha) * fkb * logFK
        x_z = np.log((np.sqrt(1 - 2 * p * z + z ** 2) + z - p) / (1 - p))
        part1_num = alpha
        part1_den = fkb * (1 + ((((1 - B) ** 2) / 24) * (logFK ** 2)) + ((((1 - B) ** 4) / 1920) * (logFK ** 4)))
        part1 = part1_num / part1_den

        part2 = z / x_z

        part3 = 1 + t * ((((1 - B) ** 2) / 24) * ((alpha ** 2) / ((F * K) ** (1 - B))) + (1 / 4) * (
                    (p * B * nu * alpha) / (fkb)) + ((2 - 3 * (p ** 2)) / (24)) * (nu ** 2))
        vol_black = part1 * part2 * part3  # num/den
    else:
        vol_black = np.NaN

    return vol_black


def objective_func_sabr(params, Beta, strikes, mkt_vols, fwd, tenor) -> np.float64:
    model_output = []
    for strike in strikes:
        model_ivol = SABR(alpha=params[0], p=params[1], nu=params[2], F=fwd, K=strike, t=tenor, B=Beta)
        model_output.append(model_ivol)
    model_output = np.array(model_output, dtype=np.float64)
    # calculate the error
    error = rmse(market=mkt_vols, model=model_output)
    return error


# Alpha as ATM restriction

def sabr_atm_alpha_cubic(alpha, Beta, fwd, rho, nu, T, mkt_atm):
    """
    :param alpha:
    :param Beta:
    :param fwd:
    :param rho:
    :param nu:
    :param T:
    :param mkt_atm: atm implied vol
    :return:
    """
    a = (((1-Beta)**2) * T) / (24* (fwd**(2-2*Beta)) )
    b = ((rho*Beta*nu*T)/(4* (fwd**(1-Beta))))
    c = 1 + T*(nu**2)*((2-3*(rho**2) ))/(24)
    eqn = a*(alpha**3) + b*(alpha**2) + c*(alpha) - mkt_atm*(fwd**(1-Beta))
    return eqn

def object_func_alpha(param, rho, nu, Beta, fwd, mkt_atm, T):
    """
    params: alpha
    """
    # want to minimise the sabr cubic
    cubic_result = sabr_atm_alpha_cubic(alpha=param, rho=rho, nu=nu, Beta=Beta, fwd=fwd, T=T, mkt_atm=mkt_atm)
    return (10000*cubic_result)**2


def objective_func_sabr_atm(params, Beta, strikes, mkt_vols, fwd, tenor, atm_vol, atm_strike, a_guess, a_bnds):
    model_output = []

    global atm_alpha_res  # will have to figure out how to access this...
    # Find Roots of Quadratic for Alpha: which is now a function of params
    alpha_guess = a_guess
    #alpha_bnds = ((0.01, 1),)# Annoying Scipy dimensions handling...
    atm_alpha_res = scipy.optimize.minimize(object_func_alpha, x0=alpha_guess,
                                            args=(params[0], params[1], Beta, fwd, atm_vol, tenor),
                                            bounds=a_bnds
                                            )
    atm_alpha = atm_alpha_res.x
    for strike in strikes:
        model_ivol = SABR(alpha=atm_alpha, p=params[0], nu=params[1], F=fwd, K=strike, t=tenor, B=Beta)
        model_output.append(model_ivol)
    model_output = np.array(model_output, dtype=np.float64)
    # calculate the error
    error = rmse(market=mkt_vols, model=model_output)
    return error