""""
 INCISO (a)

 * PICO DE BRAGG.
   La dosis depositada por unidad de profundidad cumple  D(z) ~ -dE/dx.
   Como  -dE/dx ~ 1/beta^2, un proton rapido (beta alto) deposita POCA energia
   => meseta de entrada baja. A medida que se frena, beta -> 0 y -dE/dx DIVERGE,
   concentrando casi toda la energia restante en un MAXIMO muy localizado al
   final del recorrido (el pico de Bragg); una vez agotado el proton la dosis
   cae a cero abruptamente. Esa forma "meseta + pico distal + caida" es lo que
   permite concentrar la dosis en el tumor minimizando el dano al tejido sano.

 * ECUACION DE BETHE-BLOCH.
     -dE/dx = K z^2 (Z/A) (1/beta^2) [ 0.5*ln(2 me c^2 beta^2 gamma^2 Tmax/I^2) - beta^2 ]
   Identificacion de simbolos:
     K      = 4 pi NA re^2 me c^2 = 0.307075 MeV cm^2/mol   (constante universal)
     z      = numero de carga del proyectil  (proton: z = 1)
     Z, A   = numero y masa atomicos del medio (agua: Z/A = 0.555 mol/g)
     beta   = v/c  y  gamma = factor de Lorentz, ambos DEL PROTON
     me c^2 = 0.511 MeV  (energia en reposo del electron)
     Tmax   = 2 me c^2 beta^2 gamma^2 / (1 + 2 gamma me/M + (me/M)^2)
              (maxima energia transferible a un electron en una sola colision)
     I      = potencial medio de excitacion del medio (agua: I ~ 75 eV); es la
              energia promedio transferida por colision a los electrones atomicos.
   ORIGEN DE z^2 : la fuerza de Coulomb sobre los electrones del medio es ~ z, el
     impulso transferido dp ~ z, y la energia cedida ~ dp^2 ~ z^2 (no depende del
     signo de la carga a este orden).
   ORIGEN DE 1/beta^2 : un proton mas lento pasa MAS TIEMPO (~1/v) en la vecindad
     de cada electron, por lo que dp ~ 1/v y la energia ~ dp^2 ~ 1/v^2 ~ 1/beta^2.
     Este factor es justamente lo que hace divergir la deposicion al frenarse el
     proton, dando origen al pico de Bragg.

 * ENERGY STRAGGLING (aproximacion gaussiana de Bohr).
   La perdida de energia es ESTADISTICA (el numero de colisiones y la energia
   transferida en cada una fluctuan). La varianza de la perdida en un paso dx es
     sigma_E^2 = 4 pi re^2 (me c^2)^2 Ne (z^2/beta^2) dx,
   con Ne = rho NA Z/A la densidad electronica del medio (agua: 3.34e23 /cm^3).
   Acumulada a lo largo de la traza produce la dispersion de alcances (range
   straggling) que ensancha y baja el pico de Bragg.

 * DISPERSION COULOMBIANA MULTIPLE (aproximacion de Highland).
   Desviacion angular acumulada por colisiones elasticas con los NUCLEOS:
     theta0 = (13.6 MeV / (beta c p)) z sqrt(dx/X0) [1 + 0.038 ln(dx/X0)],
   con p el momento del proton y X0 la longitud de radiacion del medio. Es la
   causa del ensanchamiento LATERAL del haz (penumbra), complementario al
   straggling longitudinal de energia/rango.
"""

import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import quad

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figuras")
os.makedirs(OUTDIR, exist_ok=True)

K=0.307075; MEC2=0.510998950; MPC2=938.27208816      
RE=2.8179403262e-13; NA=6.02214076e23                
ZA=0.55509; RHO=1.0; I_eV=75.0; I=I_eV*1e-6          
NE=RHO*NA*ZA; Z=1                                     

def beta2_gamma(E):
    gamma=1.0+E/MPC2; beta2=1.0-1.0/gamma**2
    return beta2,gamma,beta2*gamma**2
def T_max(E):
    beta2,gamma,_=beta2_gamma(E); me_M=MEC2/MPC2
    return 2.0*MEC2*beta2*gamma**2/(1.0+2.0*gamma*me_M+me_M**2)

def stopping_power_mass(E):            
    E=np.asarray(E,float); beta2,gamma,b2g2=beta2_gamma(E); Tmax=T_max(E)
    return K*Z**2*ZA/beta2*(0.5*np.log(2.0*MEC2*b2g2*Tmax/I**2)-beta2)
def stopping_power_linear(E):          
    return RHO*stopping_power_mass(E)

E_MIN=1.0; R_RESIDUAL_CM=0.00237
def csda_range_cm(E0,e_min=E_MIN):
    val,_=quad(lambda E:1.0/stopping_power_linear(E),e_min,E0,limit=400)
    return val+R_RESIDUAL_CM

OMEGA0=4.0*np.pi*RE**2*MEC2**2*NE*Z**2              
def sigma_E2(E,dx_cm):
    beta2,_,_=beta2_gamma(E); return OMEGA0*dx_cm/beta2
def sigma_R_theory_cm(E0,e_min=E_MIN):
    def f(E):
        beta2,_,_=beta2_gamma(E); return (OMEGA0/beta2)/stopping_power_linear(E)**3
    var,_=quad(f,e_min,E0,limit=400); return np.sqrt(var)

PSTAR_mm={50:22.27,150:157.70,250:379.40}            
def part_b():
    print("\n"+"="*60); print(" (b) RANGO CSDA  Bethe-Bloch vs NIST PSTAR (agua)"); print("="*60)
    print(f"{'E0[MeV]':>8} | {'R_BB[mm]':>9} | {'PSTAR[mm]':>10} | {'err%':>7}"); print("-"*60)
    rows=[]
    for E0 in (50,150,250):
        R=csda_range_cm(E0)*10.0; ref=PSTAR_mm[E0]; err=100*(R-ref)/ref
        rows.append((E0,R,ref,err)); print(f"{E0:>8} | {R:>9.2f} | {ref:>10.2f} | {err:>+7.2f}")
    return rows

def monte_carlo(E0=150.0,N=10_000,dx_mm=0.1,straggling=False,seed=1):
    """Transporta N protones en pasos dx restando dE=(-dE/dx)dx (con o sin
       fluctuacion gaussiana). Devuelve (z[mm], dosis[MeV/bin], rangos[mm])."""
    rng=np.random.default_rng(seed); dx=dx_mm/10.0
    nbins=int(np.ceil(csda_range_cm(E0)/dx))+600
    dose=np.zeros(nbins); E=np.full(N,float(E0)); alive=np.ones(N,bool)
    ranges=np.zeros(N); step=0
    while alive.any() and step<nbins:
        Ea=E[alive]; dE=stopping_power_linear(Ea)*dx               
        if straggling:                                             
            dE=dE+rng.normal(0,1,Ea.shape)*np.sqrt(sigma_E2(Ea,dx))
        will_stop=(Ea-dE)<=E_MIN
        dE=np.where(will_stop,Ea,dE)                               
        dose[step]+=dE.sum(); E[alive]=Ea-dE
        ia=np.flatnonzero(alive); sn=ia[will_stop]
        ranges[sn]=(step+1)*dx_mm; alive[sn]=False; step+=1
    if alive.any(): ranges[alive]=step*dx_mm
    z=(np.arange(nbins)+0.5)*dx_mm
    return z,dose,ranges

def main():
    rows=part_b()
    Egrid=np.linspace(1,250,500); Smass=stopping_power_mass(Egrid); dx_mm=0.1
    z,dose_c,_=monte_carlo(150.0,straggling=False,seed=1); Dz_c=dose_c/dx_mm  
    z2,dose_d,ranges_d=monte_carlo(150.0,straggling=True,seed=1); Dz_d=dose_d/dx_mm 
    R150=csda_range_cm(150.0)*10.0
    peak_c=z[np.argmax(Dz_c)]; peak_d=z2[np.argmax(Dz_d)]
    sigma_R=ranges_d.std(); mean_R=ranges_d.mean(); sigma_R_th=sigma_R_theory_cm(150.0)*10.0
    print("\n"+"="*60); print(" (c)-(d) PICO DE BRAGG  (E0=150 MeV, 1e4 protones)"); print("="*60)
    print(f"R_CSDA (integral)            = {R150:7.2f} mm")
    print(f"Pico de Bragg SIN straggling = {peak_c:7.2f} mm   (debe ~ R_CSDA)")
    print(f"Pico de Bragg CON straggling = {peak_d:7.2f} mm")
    print(f"Rango medio (con stragg.)    = {mean_R:7.2f} mm   (debe ~ R_CSDA)")
    print(f"sigma_R Monte Carlo          = {sigma_R:7.3f} mm  ({100*sigma_R/mean_R:.2f} % del rango)")
    print(f"sigma_R teorico (integral)   = {sigma_R_th:7.3f} mm  (verificacion)")
    ent=Dz_c[(z>2)&(z<5)].mean()
    print(f"Razon pico/entrada sin str.  = {Dz_c.max()/ent:7.1f}")
    print(f"Razon pico/entrada con str.  = {Dz_d.max()/ent:7.1f}")
    print(f"Energia depositada/N         = {dose_c.sum()/10000:7.2f} MeV (debe ~150)")

    fig,ax=plt.subplots(figsize=(7,4.6)); ax.loglog(Egrid,Smass,lw=2,color="#1f6feb")
    ax.set_xlabel("Energia cinetica E [MeV]"); ax.set_ylabel(r"$-dE/\rho dx$ [MeV cm$^2$/g]")
    ax.set_title("Bethe-Bloch: protones en agua (I=75 eV)"); ax.grid(True,which="both",alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,"fig_p2_stopping_power.png"),dpi=140)

    E0s=np.array([50,150,250]); Rbb=[r[1] for r in rows]; Rps=[r[2] for r in rows]
    x=np.arange(3); w=.35; fig,ax=plt.subplots(figsize=(7,4.4))
    ax.bar(x-w/2,Rbb,w,label="Bethe-Bloch",color="#1f6feb"); ax.bar(x+w/2,Rps,w,label="NIST PSTAR",color="#f0883e")
    for xi,a,b in zip(x,Rbb,Rps):
        ax.text(xi-w/2,a+4,f"{a:.1f}",ha="center",fontsize=8); ax.text(xi+w/2,b+4,f"{b:.1f}",ha="center",fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"{e} MeV" for e in E0s]); ax.set_ylabel("Rango CSDA [mm]")
    ax.set_title("Rango CSDA en agua"); ax.legend(); ax.grid(True,axis="y",alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,"fig_p2_csda_vs_pstar.png"),dpi=140)

    fig,ax=plt.subplots(figsize=(8,5)); m=z<(R150+15)
    ax.plot(z[m],Dz_c[m]/Dz_c.max(),lw=2,color="#1f6feb",label="Sin straggling (c)")
    ax.plot(z2[m],Dz_d[m]/Dz_c.max(),lw=2,color="#d62728",label="Con straggling (d)")
    ax.axvline(R150,ls="--",color="k",alpha=.7,label=f"$R_{{CSDA}}$={R150:.1f} mm")
    ax.set_xlabel("Profundidad z [mm]"); ax.set_ylabel("Dosis relativa D(z)/D$_{max}$")
    ax.set_title("Pico de Bragg - 1e4 protones de 150 MeV en agua"); ax.legend(); ax.grid(True,alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,"fig_p2_bragg_peak.png"),dpi=140)

    fig,ax=plt.subplots(figsize=(7,4.4)); ax.hist(ranges_d,bins=60,color="#d62728",alpha=.8,density=True)
    ax.axvline(mean_R,color="k",ls="--",label=f"media={mean_R:.1f} mm")
    ax.axvspan(mean_R-sigma_R,mean_R+sigma_R,color="gray",alpha=.25,label=fr"$\sigma_R$={sigma_R:.2f} mm")
    ax.set_xlabel("Rango individual [mm]"); ax.set_ylabel("Densidad"); ax.set_title("Range straggling (E0=150 MeV)")
    ax.legend(); ax.grid(True,alpha=.3); fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,"fig_p2_range_straggling.png"),dpi=140)
    print(f"\nFiguras guardadas en: {OUTDIR}")

# RESULTADOS DE REFERENCIA (de una ejecucion) E INTERPRETACION

# (b) Rango CSDA vs NIST PSTAR (agua):
#       E0=50  MeV -> 22.22 mm  (PSTAR 22.27,  -0.22 %)
#       E0=150 MeV -> 157.67 mm (PSTAR 157.70, -0.02 %)
#       E0=250 MeV -> 379.31 mm (PSTAR 379.40, -0.02 %)
# (c) Pico de Bragg (150 MeV) sin straggling: 157.85 mm ~ R_CSDA (157.67). OK.
# (d) Con straggling: pico 154.25 mm; rango medio 157.89 mm (~R_CSDA => INSESGADO);
#     sigma_R = 3.64 mm (Monte Carlo) que COINCIDE con el valor teorico 3.66 mm.
#     La razon pico/entrada cae de ~27 a ~5. Conservacion de energia: 150.0 MeV/p.
# Nota: sigma_R ~2.3 % es mayor que el ~1.1 % clinico empirico porque la formula
# de Bohr del enunciado lleva el factor 1/beta^2; el acuerdo MC<->teoria valida la
# implementacion. RELEVANCIA CLINICA: sigma_R fija el MARGEN DE SEGURIDAD DISTAL
# cuando hay un organo critico detras del tumor (no se puede colocar el pico de
# Bragg demasiado cerca por la incertidumbre de rango).

if __name__ == "__main__":
    main()
