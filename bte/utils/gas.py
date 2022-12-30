import math

def get_gamma(ck):
    get_gamma = (ck+5)/(ck+3)
    return get_gamma

def get_potential(omega):
    if(omega==0.5):
        alpha=1.0
    else:
        eta=4.0/(2.0*omega-1.0)+1.0
        alpha=(eta-5.0)/(eta-1.0)
    return alpha

def get_mu(alpha,omega,kn):
    get_mu = 5*(alpha+1)*(alpha+2)*math.sqrt(math.pi)/(4*alpha*(5-2*omega)*(7-2*omega))*kn
    return  get_mu

def get_kn_bzm(alpha,mu_ref):
    kn_bzm=64*math.sqrt(2.0)**alpha/5.0*math.gamma((alpha+3)/2)*math.gamma(2.0)*math.sqrt(math.pi)*mu_ref
    return kn_bzm
    