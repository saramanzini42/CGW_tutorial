import functools
import astropy.constants as astropy_const
import astropy.units as u
import numpy as np
import jax
import jax.numpy as jnp
import interpax
from . import matrix
from . import const
from jax.experimental.ode import odeint
import diffrax
jax.config.update("jax_enable_x64", True)

Msun = astropy_const.M_sun.to(u.kg).value
c = astropy_const.c.to(u.m / u.s).value
G = astropy_const.G.to(u.m**3 / (u.kg * u.s**2)).value
MGsunsec = Msun*G / c**3
kpc = 1.0* astropy_const.kpc.to(u.m).value/c
pc =  1.0* astropy_const.pc.to(u.m).value/c


def t_p(t, Lp, cos_mu):
    tau_p = Lp * kpc * (1 - cos_mu)
    tp = t - tau_p
    return tp


def xdot(x, M, nu, e, order = 1):
    e2 = e**2
    e4 = e2*e2
    e6 = e4*e2
    term1 =2/3*(nu*(96 + 292*e2 + 37*e4)*x**(5))/(5*M*(1 - e2)**(7/2))
    term2 = ((nu*(16*(743 + 924*nu) + e6*(6931 + 2072*nu) + 14*e4*(7079 + 3690*nu) + 8*e2*(15411 + 11158*nu)))*x**6)/(420*((1 - e2)**(9/2)*M))
        
    return term1 - order*term2

def edot(x, M, nu, e, order=1):
    e2 = e*e
    e4 = e2*e2
    e6 = e4*e2
    term1 = -(e*nu*(304 + 121*e2)*x**(4))/(15*M*(1 - e2)**(5/2)) 
    term2 = ((e*nu*(e4*(94887 + 19768*nu) + 12*e2*(38698 + 21427*nu) + 8*(20547 + 24556*nu))*x**5)/(2520*(1 - e2)**(7/2)*M))
    
    return term1 + order*term2
    
   
def gammadot(x, M, nu, e, order=1):
   
    e2 = e**2
    term1 =  3*x**(5/2)/(M*(1 - e2))   
    term2 = ((18 - 21*e2 - 28*nu - 2*e2*nu)*x**(7/2))/(4*(-1 + e2)**2*M)
    return term1 + order*term2


def ode_system_fast(y, t, M, nu):
    e, x, gamma = y
    
    dedt = edot(x, M, nu, e, 1)
    dxdt = xdot(x, M, nu, e, 1)
    dgammadt = gammadot(x, M, nu, e, 1)

    return jnp.array([-dedt, -dxdt, -dgammadt])


# Diffrax requires a VectorField class with signature (t, y, args)
class ODESystem(diffrax.AbstractTerm):
    def vf(self, t, y, args):
        M, nu = args
        e, x, gamma = y

        dedt = edot(x, M, nu, e, 1)
        dxdt = xdot(x, M, nu, e, 1)
        dgammadt = gammadot(x, M, nu, e, 1)

        return jnp.array([dedt, dxdt, dgammadt])

    def contr(self, t0, t1):
        return t1 - t0

    def prod(self, vf, control):
        return vf * control


def xi_t(M, x, xi0, omega_r, nu, e, toas):
    xi =xi0 + jnp.linspace(0, omega_r*(jnp.max(toas) - jnp.min(toas)+1e9), int(len(toas)))
    
    
    p = (1-e**2)/x + (1/3)*(nu +e**2*(6-nu))
    
    a = 2*(6 + 2*p + nu - e**2*(6 + nu))*(jnp.unwrap(jnp.arctan(jnp.tan(xi/2)*(jnp.sqrt((1-e)/(1+e)))), period = jnp.pi))/(1 - e**2)**(3/2)
    
    b = -(72*(jnp.unwrap(jnp.arctan((jnp.sqrt(-(-6 + 2*p + nu + e*(6 - e*nu))/(6 - 2*p - nu + e*(6 + e*nu)))*jnp.tan(xi/2))), period = jnp.pi))/jnp.sqrt(((-6 + 2*p + nu + e*(6 - e*nu))*(-6 + 2*p + nu - e*(6 + e*nu)))))
    
    c = -(e*(2*p + nu - e**2*nu)*jnp.sin(xi))/((1 - e**2)*(1 + e*jnp.cos(xi)))
    
    t =(((2*M*p**(5/2))/(2*p + nu - nu*e**2)**2)*(a+b+c)) 
    interp = interpax.Interpolator1D(t-t[0], xi, method = 'cubic')
    return interp(toas-toas[0])

def res_analit_plus(e, gamma, t, Mc, D, xi, iota, omega_phi, omega_r): 
    coeff = (Mc**(5/3)*jnp.sqrt(1-e**2))*omega_phi**(2/3)/(D*omega_r)
    a = (e+2*jnp.cos(xi))*jnp.sin(xi)
    b = (jnp.cos(2*xi) + e*jnp.cos(xi))
    c = (e*jnp.sin(xi))
    den = (1+e*jnp.cos(xi))
    return coeff/den*((1+jnp.cos(iota)**2)*(a*jnp.cos(2*gamma)+b*jnp.sin(2*gamma)) + jnp.sin(iota)**2*c)

   
def res_analit_cross(e, gamma, t, Mc, D, xi, iota, omega_phi, omega_r):
    coeff = (Mc**(5/3)*jnp.sqrt(1-e**2))*omega_phi**(2/3)/(D*omega_r)
    a = (e+2*jnp.cos(xi))*jnp.sin(xi)
    b = (jnp.cos(2*xi) + e*jnp.cos(xi))
    den = (1+e*jnp.cos(xi))
    return coeff/den*(2*jnp.cos(iota))*(a*jnp.sin(2*gamma)-b*jnp.cos(2*gamma))


def create_gw_antenna_pattern(pos, gwtheta, gwphi):
    """
    :return: (fplus, fcross, cosMu), where fplus and fcross
             are the plus and cross antenna pattern functions
             and cosMu is the cosine of the angle between the
             pulsar and the GW source.
    """
    sin_gwphi = jnp.sin(gwphi)
    cos_gwphi = jnp.cos(gwphi)
    sin_gwtheta = jnp.sin(gwtheta)
    cos_gwtheta = jnp.cos(gwtheta)
    
    # use definition from Sesana et al 2010 and Ellis et al 2012
    m = jnp.array([-sin_gwphi, cos_gwphi, 0.0])
    n = jnp.array([-cos_gwtheta * cos_gwphi, -cos_gwtheta * sin_gwphi, sin_gwtheta])
    omhat = jnp.array([-sin_gwtheta * cos_gwphi, -sin_gwtheta* sin_gwphi, -cos_gwtheta])

    fplus = 0.5 * (jnp.dot(m, pos) ** 2 - jnp.dot(n, pos) ** 2) / (1 + jnp.dot(omhat, pos))
    fcross = (jnp.dot(m, pos) * jnp.dot(n, pos)) / (1 +jnp.dot(omhat, pos))
    
    cosMu = -jnp.dot(omhat, pos)

    return fplus, fcross, cosMu



def make_delay_eccentric():
    #return function to compute residuals
    ode_term = ODESystem()

    def eccentric_delay(toas, pos, psrdist, cos_gwtheta, gwphi, log10_M, nu,
                                   log10_dist, log10_Forb, cos_inc, psi, gamma0,
                                   xi0, xi0p, e0):
        tref = 0  # MJD J2000 in seconds
        #return residuals
        # --- Initialization (from __init__ method) ---
        M = 10**log10_M * MGsunsec
        D = (10**log10_dist) * (1e6 * pc)
        iota = jnp.arccos(cos_inc)
        F0 = 10**log10_Forb
        x0 = (M * 2 * jnp.pi * F0)**(2/3)
        Mc =  M* nu**(3/5)

        Fplus, Fcross, cos_mu = create_gw_antenna_pattern(pos, jnp.arccos(cos_gwtheta), gwphi)

        # --- Evolve dynamics approximately ---
        def ObservedDynamics(dt, e0, x0, gamma0, xi0):
            d_e0, d_x0, d_gamma0 = ode_system_fast(jnp.array([e0, x0, gamma0]), 0.0, M, nu)
            dt = toas - toas[0]
            omega_phi0 = x0**(3/2) / M
            p0 = (1 - e0**2) / x0 + (1/3) * (nu + e0**2 * (6 - nu))
            omega_r0 = (1 - e0**2)**(3/2) / (M * p0**(3/2)) * (1 + ((1 - e0**2) * (-6 + nu)) / (2 * p0))
            #print('omega_r0 new = ', omega_r0)
            
            ### Need forward evolution, default derivatives are backward
            e = e0 - d_e0 * dt
            x = x0 - d_x0 * dt
            xi = xi_t(M, x, xi0, omega_r0, nu, e, toas)
            gamma = gamma0 + (omega_phi0 - omega_r0)*dt
            
            return e, x, xi, gamma
        e, x, xi, gamma = ObservedDynamics(toas, e0, x0, gamma0, xi0)


        def res_internal(current_t, e_oft, x_oft, xi_oft, gamma_oft):

            #omega_phi = x_oft**(3/2) / M 
            #omega_r = omega_phi*(1 - x_oft*3/(1 - e_oft**2))
            p = (1 - e_oft**2) / x_oft + (1/3) * (nu + e_oft**2 * (6 - nu))
            omega_r = (1 - e_oft**2)**(3/2) / (M * p**(3/2)) * (1 + ((1 - e_oft**2) * (-6 + nu)) / (2 * p))
            omega_phi = x_oft**(3/2) / M
            #print('omega_r new = ', omega_r)
            rplus = res_analit_plus(e_oft, gamma_oft, current_t, Mc, D, xi_oft, iota, omega_phi, omega_r)
            rcross = res_analit_cross(e_oft, gamma_oft, current_t, Mc, D, xi_oft, iota, omega_phi, omega_r)

            return (Fplus * jnp.cos(2*psi) + Fcross * jnp.sin(2*psi)) * rplus - \
                    (Fplus * jnp.sin(2*psi) - Fcross * jnp.cos(2*psi)) * rcross

        ET = res_internal(toas, e, x, xi, gamma)

        ### now Pulsar term
        ### integrate backwards
        tpulsar = t_p(toas.min(), psrdist, cos_mu) ### time
        tp_arr = jnp.linspace(tpulsar, toas.min(), int(1e4)) ### CHECK! how does it depend on num of points??? 
        y0 = jnp.array([e0, x0, gamma0])
        #y_back = odeint(ode_system_fast, y0, tp_arr, M, nu)
        #ep_ar, xp_ar, gammap_ar = y_back.T
        #ep0, xp0, gammap0 = ep_ar[-1], xp_ar[-1], gammap_ar[-1]
        solver = diffrax.Tsit5()

            # Adaptive step size controller
        stepsize_controller = diffrax.PIDController(
            rtol=1e-4,
            atol=1e-6,
            dtmin=1e-12,
            dtmax=(toas.min() - tpulsar) / 10.0  # Max step = 10% of interval
        )


        # Solve backwards: from toas.min() to tpulsar
        solution = diffrax.diffeqsolve(
            ode_term,
            solver,
            t0=toas.min(),
            t1=tpulsar,
            dt0=-(toas.min() - tpulsar) / 100.0,  # Initial step (negative = backwards)
            y0=y0,
            args=(M, nu),
            stepsize_controller=stepsize_controller,
            saveat=diffrax.SaveAt(t1=True),  # Only save final state
            max_steps=6048,
            throw=False
            )


        # Extract final state (at tpulsar)
        ep0, xp0, gammap0 = solution.ys[0]
        #print(ep0)
        tp_arr = t_p(toas, psrdist, cos_mu) 
        ep, xp, xip, gammap = ObservedDynamics(toas, ep0, xp0, gammap0, xi0p)
        PT = res_internal(toas, ep, xp, xip, gammap) 

        return ET - PT
    
    return eccentric_delay