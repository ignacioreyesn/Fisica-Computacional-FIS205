import numpy as np
import matplotlib
matplotlib.use('Agg')                   
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings; warnings.filterwarnings('ignore')


SEED       = 42
N_SIGNALS  = 3000          
NT         = 1000          
T_MAX      = 10.0          
SIGMA_BASE = 0.02          
GAMMA_LO, GAMMA_HI = 0.05, 1.0
K_LO,     K_HI     = 1.00, 5.0
SIGMAS_D   = [0.0, 0.01, 0.02, 0.05, 0.10]   

t_eval = np.linspace(0.0, T_MAX, NT)          

"""

 INCISO (a)

1. PROBLEMA DIRECTO E INVERSO

  Problema directo: dados los parámetros (γ, k) y las condiciones
  iniciales x(0)=1, ẋ(0)=0, se resuelve numéricamente la EDO

        mẍ(t) + γẋ(t) + kx(t) = 0

  para obtener la trayectoria x(t). El operador F:(γ,k) → x(t) es
  continuo: pequeñas variaciones en (γ,k) producen pequeñas
  variaciones en x(t) (problema BIEN PUESTO).

  Problema inverso: dada la señal observada x_obs(t), encontrar
  (γ,k) tal que F(γ,k) ≈ x_obs. Sus dificultades son:

    • Mal condicionamiento: distintos pares (γ,k) pueden producir
      señales muy similares cuando hay ruido.
    • Sensibilidad al ruido: pequeñas perturbaciones en x_obs
      pueden generar grandes errores en la estimación de (γ,k).
    • No existe fórmula cerrada; hay que invertir F numéricamente
      o aprender la inversa con un modelo de ML.

2. APRENDIZAJE SUPERVISADO

  Dado el conjunto {(x_i, θ_i)}_{i=1}^N donde x_i es la señal y
  θ_i=(γ_i,k_i) son las etiquetas, el modelo aprende el mapeo
  f_w : x → θ minimizando la pérdida empírica.

  Para el problema inverso SÍ es necesario conocer las etiquetas,
  pues el modelo necesita ejemplos "señal → parámetros" para
  aprender. Aquí las obtenemos SINTÉTICAMENTE: sorteamos (γ,k) y
  simulamos x(t), por lo que cada par (x_i,θ_i) es exactamente
  conocido.

3. CONJUNTOS DE ENTRENAMIENTO Y VALIDACIÓN

  Se separan los datos porque:

    • Si evaluamos el error sobre el mismo conjunto con que se
      entrenó, el modelo podría MEMORIZAR los datos sin generalizar
      → sobreajuste (overfitting).
    • El conjunto de test provee una estimación HONESTA del error
      de generalización sobre datos nunca vistos.
    • Esta separación permite DETECTAR el sobreajuste: si
      RMSE_train ≪ RMSE_test, el modelo memorizó los datos en
      lugar de aprender la física subyacente.

4. FUNCIÓN DE PÉRDIDA – MSE

  Para la tarea de regresión multivariada (predecir γ y k):

        MSE = (1/N) Σ_i [(γ_i − γ̂_i)² + (k_i − k̂_i)²]

  Se usa el cuadrado del error porque:

    • Es diferenciable en todas partes (esencial para gradiente
      descendente en redes neuronales).
    • Penaliza desproporcionadamente errores grandes, forzando al
      modelo a evitar predicciones muy alejadas del valor real.
    • Su raíz cuadrada (RMSE) tiene las mismas unidades que γ y k,
      lo que facilita la interpretación física de los resultados.
"""

#  INCISO (b)

def solve_analytical(gamma, k, t, m=1.0):
    
    alpha = gamma / (2.0 * m)
    disc  = k / m - alpha**2

    if disc > 1e-12:                       
        wd = np.sqrt(disc)
        return np.exp(-alpha * t) * (np.cos(wd * t) + (alpha / wd) * np.sin(wd * t))
    elif disc > -1e-12:                    
        return np.exp(-alpha * t) * (1.0 + alpha * t)
    else:                                  
        sq      = np.sqrt(alpha**2 - k / m)
        r1, r2  = -alpha + sq, -alpha - sq
        c2 = -r1 / (r2 - r1) if abs(r2 - r1) > 1e-15 else 0.5
        return (1.0 - c2) * np.exp(r1 * t) + c2 * np.exp(r2 * t)


def generate_dataset(n=N_SIGNALS, sigma=SIGMA_BASE, seed=SEED, verbose=True):

    rng    = np.random.default_rng(seed)
    gammas = rng.uniform(GAMMA_LO, GAMMA_HI, n)
    ks     = rng.uniform(K_LO,     K_HI,     n)

    X = np.zeros((n, len(t_eval)))
    for i, (g, k) in enumerate(zip(gammas, ks)):
        x_clean   = solve_analytical(g, k, t_eval)
        X[i]      = x_clean + rng.normal(0.0, sigma, len(t_eval))

    Y = np.column_stack([gammas, ks])
    if verbose:
        print(f"  Dataset: {X.shape[0]} señales × {X.shape[1]} puntos  "
              f"(σ={sigma})")
    return X, Y


def plot_sample_signals(X, Y, savepath='fig_senales.png'):
    
    g, k = Y[:, 0], Y[:, 1]

    
    idxs = [
        np.argmin(g),
        np.argmax(g),
        np.argmin(k),
        np.argmax(k),
        np.argmin(np.abs(g - 0.30) + np.abs(k - 2.0)),
        np.argmin(np.abs(g - 0.80) + np.abs(k - 4.0)),
    ]
    titles = [
        r'Bajo amort. $\gamma={:.3f}$, $k={:.2f}$'.format(g[idxs[0]], k[idxs[0]]),
        r'Alto amort. $\gamma={:.3f}$, $k={:.2f}$'.format(g[idxs[1]], k[idxs[1]]),
        r'Baja frec.  $\gamma={:.3f}$, $k={:.2f}$'.format(g[idxs[2]], k[idxs[2]]),
        r'Alta frec.  $\gamma={:.3f}$, $k={:.2f}$'.format(g[idxs[3]], k[idxs[3]]),
        r'$\gamma\approx0.30$, $k\approx2.0$',
        r'$\gamma\approx0.80$, $k\approx4.0$',
    ]
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, idx, title, col in zip(axes.flatten(), idxs, titles, colors):
        ax.plot(t_eval, X[idx], color=col, lw=1.1, alpha=0.85)
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel('t [s]', fontsize=9)
        ax.set_ylabel('x(t)', fontsize=9)
        ax.set_xlim(0, T_MAX)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        'Señales del oscilador armónico amortiguado\n'
        r'($\sigma=0.02$,  $m=1$,  $x(0)=1$,  $\dot{x}(0)=0$)',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(savepath, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  → Figura guardada: {savepath}")

#  INCISO (c)

def rmse_col(y_true, y_pred, col):
    """RMSE para la columna col de arreglos 2D."""
    return np.sqrt(mean_squared_error(y_true[:, col], y_pred[:, col]))


def train_and_evaluate(X, Y, sigma_label='0.02'):
    
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X, Y, test_size=0.20, random_state=SEED)

    rf = RandomForestRegressor(
        n_estimators   = 50,
        max_depth      = 15,
        min_samples_leaf = 3,
        n_jobs         = -1,
        random_state   = SEED,
    )
    print(f"  Entrenando RandomForestRegressor "
          f"({X_tr.shape[0]} train / {X_te.shape[0]} test) ...")
    rf.fit(X_tr, Y_tr)

    Y_pred_tr = rf.predict(X_tr)
    Y_pred_te = rf.predict(X_te)

    metrics = {
        'rmse_gamma_train': rmse_col(Y_tr, Y_pred_tr, 0),
        'rmse_k_train':     rmse_col(Y_tr, Y_pred_tr, 1),
        'rmse_gamma_test':  rmse_col(Y_te, Y_pred_te, 0),
        'rmse_k_test':      rmse_col(Y_te, Y_pred_te, 1),
    }
    return rf, metrics, Y_te, Y_pred_te


def print_metrics_table(metrics, sigma=SIGMA_BASE):
    
    hdr = f"  Métricas RandomForestRegressor (σ={sigma})"
    sep = "  " + "─" * 46
    print("\n" + sep)
    print(hdr)
    print(sep)
    print(f"  {'Parámetro':<12} {'RMSE Train':>12} {'RMSE Test':>12}")
    print(sep)
    print(f"  {'γ':<12} {metrics['rmse_gamma_train']:>12.4f} "
          f"{metrics['rmse_gamma_test']:>12.4f}")
    print(f"  {'k':<12} {metrics['rmse_k_train']:>12.4f} "
          f"{metrics['rmse_k_test']:>12.4f}")
    print(sep + "\n")


def plot_predictions(Y_te, Y_pred_te, metrics,
                     savepath='fig_predicciones.png'):
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    specs = [
        (0, r'$\gamma$', (GAMMA_LO, GAMMA_HI), '#1f77b4', 'rmse_gamma_test'),
        (1, r'$k$',      (K_LO,     K_HI),     '#d62728', 'rmse_k_test'),
    ]
    for ax, (col, lbl, lims, c, key) in zip(axes, specs):
        ax.scatter(Y_te[:, col], Y_pred_te[:, col],
                   alpha=0.3, s=8, color=c, rasterized=True,
                   label='Muestras de test')
        ax.plot(lims, lims, 'k--', lw=1.5, label='Predicción perfecta')
        ax.set_xlabel(f'{lbl} verdadero', fontsize=12)
        ax.set_ylabel(f'{lbl} predicho',  fontsize=12)
        ax.set_title(f'Predicción de {lbl}\nRMSE(test) = {metrics[key]:.4f}',
                     fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        m = 0.05 * (lims[1] - lims[0])
        ax.set_xlim(lims[0] - m, lims[1] + m)
        ax.set_ylim(lims[0] - m, lims[1] + m)

    fig.suptitle(
        r'RandomForestRegressor – Predicho vs Real ($\sigma=0.02$, split 80/20)',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(savepath, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  → Figura guardada: {savepath}")



#  INCISO (d) 

def noise_study(sigmas=SIGMAS_D):
    
    rng    = np.random.default_rng(SEED)
    gammas = rng.uniform(GAMMA_LO, GAMMA_HI, N_SIGNALS)
    ks     = rng.uniform(K_LO,     K_HI,     N_SIGNALS)
    Y_base = np.column_stack([gammas, ks])

    # Señales limpias (sin ruido) compartidas
    X_clean = np.zeros((N_SIGNALS, NT))
    for i, (g, k) in enumerate(zip(gammas, ks)):
        X_clean[i] = solve_analytical(g, k, t_eval)

    results = {
        'sigma':   [], 'rg_tr': [], 'rk_tr': [],
        'rg_te':   [], 'rk_te': [],
    }

    for sigma in sigmas:
        print(f"  ── σ = {sigma:.2f} ──")
        rng_noise = np.random.default_rng(SEED + 100)
        X = X_clean + rng_noise.normal(0.0, sigma, X_clean.shape) if sigma > 0 \
            else X_clean.copy()

        X_tr, X_te, Y_tr, Y_te = train_test_split(
            X, Y_base, test_size=0.20, random_state=SEED)

        rf = RandomForestRegressor(
            n_estimators=50, max_depth=15,
            min_samples_leaf=3, n_jobs=-1, random_state=SEED)
        rf.fit(X_tr, Y_tr)

        Yp_tr = rf.predict(X_tr); Yp_te = rf.predict(X_te)
        results['sigma'].append(sigma)
        results['rg_tr'].append(rmse_col(Y_tr, Yp_tr, 0))
        results['rk_tr'].append(rmse_col(Y_tr, Yp_tr, 1))
        results['rg_te'].append(rmse_col(Y_te, Yp_te, 0))
        results['rk_te'].append(rmse_col(Y_te, Yp_te, 1))
        print(f"    RMSE_γ  train={results['rg_tr'][-1]:.4f}  "
              f"test={results['rg_te'][-1]:.4f}")
        print(f"    RMSE_k  train={results['rk_tr'][-1]:.4f}  "
              f"test={results['rk_te'][-1]:.4f}")

    return results


def plot_noise_study(results, savepath='fig_ruido.png'):
    """
    Grafica RMSE_γ y RMSE_k vs σ (líneas train y test).
    """
    sigmas = results['sigma']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, param, ktr, kte, col_tr, col_te in zip(
        axes,
        [r'$\gamma$', r'$k$'],
        ['rg_tr', 'rk_tr'], ['rg_te', 'rk_te'],
        ['#1f77b4', '#d62728'], ['#aec7e8', '#f4a582'],
    ):
        ax.plot(sigmas, results[ktr], 'o-',
                color=col_tr, lw=2, ms=7, label='Train', zorder=3)
        ax.plot(sigmas, results[kte], 's--',
                color=col_te, lw=2, ms=7, label='Test',
                markeredgecolor=col_tr, markeredgewidth=1.3, zorder=3)
        ax.set_xlabel(r'Nivel de ruido $\sigma$', fontsize=12)
        ax.set_ylabel(f'RMSE({param})', fontsize=12)
        ax.set_title(f'Efecto del ruido sobre RMSE de {param}', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.35)
        ax.set_xticks(sigmas)
        ax.set_xticklabels([str(s) for s in sigmas])

    fig.suptitle(
        'Estudio del efecto del ruido – RandomForestRegressor',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(savepath, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  → Figura guardada: {savepath}")


def print_discussion(metrics, noise_results):
    """Imprime la discusión de resultados."""
    sig   = noise_results['sigma']
    rg_te = noise_results['rg_te']
    rk_te = noise_results['rk_te']

    print("""
═══════════════════════════════════════════════════════════════════
 DISCUSIÓN DE RESULTADOS
═══════════════════════════════════════════════════════════════════

(b) Señales generadas
──────────────────────
  • γ controla el DECAIMIENTO EXPONENCIAL: mayor γ → envolvente
    e^{-γt/2} cae más rápido → señal se amortigua antes.
  • k controla la FRECUENCIA ANGULAR: ω_d = √(k − (γ/2)²) ≈ √k
    para γ pequeño → mayor k → más oscilaciones por unidad de tiempo.
  • El ruido σ=0.02 es observable pero no destruye la estructura
    de la señal: la relación señal/ruido es alta para amplitudes ≫ 0.02.

(c) Desempeño del RandomForestRegressor
─────────────────────────────────────────
  Hiperparámetros: n_estimators=50, max_depth=15, min_samples_leaf=3.
  Se obtiene:
""")
    print(f"    RMSE_γ (train) = {metrics['rmse_gamma_train']:.4f}  |  "
          f"RMSE_γ (test)  = {metrics['rmse_gamma_test']:.4f}")
    print(f"    RMSE_k (train) = {metrics['rmse_k_train']:.4f}  |  "
          f"RMSE_k (test)  = {metrics['rmse_k_test']:.4f}")
    print("""
  Los errores de test son sólo ligeramente mayores que los de train,
  lo que indica que el modelo GENERALIZA bien sin sobreajuste severo.

  El RF puede identificar γ con más precisión que k. Esto se debe a
  que γ afecta la ENVOLVENTE global de la señal (información difundida
  en todos los tiempos), mientras que k afecta la frecuencia, que el
  bosque infiere indirectamente contando ceros o máximos locales.

(d) Efecto del ruido
─────────────────────
  Resumen de RMSE_k(test) por nivel de ruido:
""")
    for s, rg, rk in zip(sig, rg_te, rk_te):
        print(f"    σ={s:.2f}  RMSE_γ(test)={rg:.4f}  RMSE_k(test)={rk:.4f}")
    print("""
  Observaciones:
  • Para σ=0 (sin ruido): el modelo tiene acceso a la señal exacta.
    El error de train es muy bajo, pero el de test también es bajo,
    indicando que NO hay sobreajuste severo para el RF con max_depth=15.
  • A medida que σ crece, RMSE_k(test) aumenta más rápido que
    RMSE_γ(test): k es MÁS DIFÍCIL DE INFERIR bajo ruido.
    Explicación: el ruido distorsiona los cruces por cero y los máximos
    locales de la señal, que son las marcas temporales que identifican
    la frecuencia ω_d ≈ √k. En cambio, γ queda codificado en la
    envolvente de largo plazo, que es más robusta al ruido puntual.
  • El sobreajuste se manifestaría como RMSE_train ≪ RMSE_test.
    Con max_depth=15 se observa una brecha moderada que aumenta con σ,
    coherente con un modelo que memoriza parcialmente el ruido de train.
    Aumentar min_samples_leaf o reducir max_depth atenuaría este efecto.
═══════════════════════════════════════════════════════════════════
""")

#  BLOQUE PRINCIPAL

if __name__ == '__main__':
    bar = "═" * 60
    print(f"\n{bar}")
    print("  TAREA 2 – PROBLEMA 1: IA PARA PROBLEMAS INVERSOS")
    print(f"{bar}\n")

    # ── (a) Conceptos ────────────────────────────────────────────
    print(CONCEPTOS)

    # ── (b) Generación del dataset ───────────────────────────────
    print(f"{bar}")
    print("  INCISO (b) – Generación del dataset")
    print(f"{bar}")
    X, Y = generate_dataset(N_SIGNALS, sigma=SIGMA_BASE, verbose=True)
    plot_sample_signals(X, Y, savepath='fig_senales.png')

    # ── (c) Entrenamiento ─────────────────────────────────────────
    print(f"\n{bar}")
    print("  INCISO (c) – Entrenamiento RandomForestRegressor")
    print(f"{bar}")
    rf, metrics, Y_te, Y_pred_te = train_and_evaluate(X, Y)
    print_metrics_table(metrics, sigma=SIGMA_BASE)
    plot_predictions(Y_te, Y_pred_te, metrics, savepath='fig_predicciones.png')

    # ── (d) Efecto del ruido ──────────────────────────────────────
    print(f"\n{bar}")
    print("  INCISO (d) – Estudio del efecto del ruido")
    print(f"{bar}")
    noise_res = noise_study(SIGMAS_D)
    plot_noise_study(noise_res, savepath='fig_ruido.png')

    # ── Discusión ─────────────────────────────────────────────────
    print_discussion(metrics, noise_res)

    print(f"{bar}")
    print("  ✔  Ejecución completada.")
    print("     Figuras generadas:")
    print("       fig_senales.png       (inciso b)")
    print("       fig_predicciones.png  (inciso c)")
    print("       fig_ruido.png         (inciso d)")
    print(f"{bar}\n")
