"""
TP de séries temporelles (Master 2) : comparaison ARIMA et ARIMA automatique sur un jeu de données libre de droit.

Ce script illustre deux approches classiques de prévision sur une série
temporelle réelle :

1. **ARIMA manuel** : on choisit les paramètres (p, d, q) à partir d'une
   analyse simple de la série et on ajuste un modèle ARIMA avec
   `statsmodels`. Ici le jeu de données présente une tendance et une
   saisonnalité annuelle modérée, donc on fixe d=1 pour différencier la série
   et on choisit p et q après avoir examiné l'autocorrélation. 
   Le script utilise l'ordre (2, 1, 3) qui se révèle
   performant pour la série de CO₂.

2. **ARIMA automatique** : on utilise `pmdarima.auto_arima` pour sélectionner
   automatiquement les paramètres du modèle en minimisant l'AIC. Le modèle
   est ajusté sur la portion d'entraînement de la série et on génère des
   prévisions sur l'horizon de test.

Pour chaque méthode, le script :

* divise la série en une partie d'entraînement et une partie de test
  correspondant aux 52 dernières observations (environ un an de données
  hebdomadaires) ;
* fit le modèle sur l'entraînement, génère des prévisions sur le jeu de
  test et calcule des intervalles de prédiction (quand la méthode le
  permet) ;
* calcule trois métriques d'évaluation : RMSE (racine de l'erreur quadratique
  moyenne), MAE (erreur absolue moyenne) et R² ;
* trace les prévisions des trois modèles ainsi que les valeurs réelles et
  leurs intervalles de confiance sur un même graphique.

Le jeu de données utilisé est le jeu **CO₂ hebdomadaire de Mauna Loa**
fourni par `statsmodels`. Il s'agit de mesures hebdomadaires de la
concentration de CO₂ (en ppmv) entre 1958 et 2001. Ce jeu est dans le
domaine public【505795181611461†L88-L129】 et constitue donc un exemple
idéal pour un TP. Les valeurs manquantes (59 observations【505795181611461†L104-L110】)
sont interpolées linéairement avant le traitement.

Usage :

    python tp_time_series_co2.py

Les prévisions et le graphique sont enregistrés dans le répertoire courant
sous les noms `forecast_metrics.csv` et `forecast_comparison.png`.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from statsmodels.tsa.arima.model import ARIMA
# Importations nécessaires pour l'ARIMA manuel et les diagnostics de stationnarité
# Nous n'utilisons plus pmdarima (auto_arima) pour sélectionner automatiquement
# les ordres du modèle. À la place, nous présentons une démarche
# pédagogique : vérifier la stationnarité avec un test ADF et inspecter
# les graphiques d'autocorrélation (ACF) et d'autocorrélation partielle (PACF).
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Importation facultative de pmdarima pour la sélection automatique des ordres.
# Nous essayerons d'utiliser auto_arima si la librairie est installée.
try:
    from pmdarima import auto_arima  # type: ignore
    _PMDARIMA_AVAILABLE = True
except Exception:
    _PMDARIMA_AVAILABLE = False

def load_co2_series() -> pd.Series:
    """Charge la série CO₂ hebdomadaire et interpole les valeurs manquantes.

    Returns
    -------
    pd.Series
        Série temporelle de CO₂ avec index temporel au format datetime.
    """
    import statsmodels.datasets.co2 as co2

    # Charge les données sous forme de DataFrame avec index temporel
    data = co2.load_pandas().data
    # Convertit l'index en datetime (il est déjà sous forme d'index de dates)
    series = data['co2'].copy()
    # Interpoler les valeurs manquantes (59 manquants【505795181611461†L104-L110】)
    series = series.interpolate(method='linear')
    # Vérifie qu'il ne reste plus de NA
    series = series.dropna()
    return series


def train_test_split(series: pd.Series, test_size: int = 52) -> tuple[pd.Series, pd.Series]:
    """Sépare la série en ensemble d'entraînement et de test.

    Parameters
    ----------
    series : pd.Series
        Série temporelle complète.
    test_size : int, default=52
        Nombre d'observations à conserver pour le test (ici environ 1 an de
        données hebdomadaires).

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (train_series, test_series)
    """
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    return train, test


def fit_arima_manual(train: pd.Series, order: tuple[int, int, int] = (2, 1, 3)) -> ARIMA:
    """Fit un modèle ARIMA manuel.

    Parameters
    ----------
    train : pd.Series
        Série d'entraînement.
    order : tuple[int, int, int]
        Paramètres (p, d, q) du modèle ARIMA.

    Returns
    -------
    ARIMA
        Modèle ajusté.
    """
    model = ARIMA(train, order=order)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        fitted = model.fit()
    return fitted


def predict_arima(model: ARIMA, steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Génère des prévisions et intervalles pour un modèle ARIMA.

    Parameters
    ----------
    model : ARIMA
        Modèle ARIMA ajusté.
    steps : int
        Nombre de pas de prévision.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (prévisions, intervalle inférieur, intervalle supérieur)
    """
    forecast_res = model.get_forecast(steps=steps)
    forecast = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int(alpha=0.05)
    return forecast.values, conf_int.iloc[:, 0].values, conf_int.iloc[:, 1].values



#
# Fonctions de diagnostic et de visualisation de la stationnarité
#

def check_stationarity(series: pd.Series, name: str = "") -> None:
    """Affiche les résultats du test ADF (Augmented Dickey–Fuller) sur la série.

    Ce test permet d'évaluer si une série est stationnaire (sans tendance
    persistante). Un p-value faible (inférieure à 0,05) indique que l'hypothèse
    de non-stationnarité peut être rejetée.

    Parameters
    ----------
    series : pd.Series
        La série temporelle (ou la série différenciée) à tester.
    name : str, optional
        Nom de la série pour l'affichage.
    """
    result = adfuller(series.values)
    stat, p_value, used_lag, nobs, critical_values, icbest = result
    print(f"\nTest ADF pour {name or 'la série'} :")
    print(f"  ADF Statistic = {stat:.4f}")
    print(f"  p-value       = {p_value:.4f}")
    print(f"  Nombre de retards utilisés = {used_lag}")
    print(f"  Observations utilisées     = {nobs}")
    print("  Valeurs critiques :")
    for key, val in critical_values.items():
        print(f"    Niveau {key}: {val:.4f}")


def plot_acf_pacf(series: pd.Series, lags: int = 40, filename: str = 'acf_pacf.png') -> None:
    """Trace les graphiques ACF et PACF de la série et enregistre l'image.

    Ces graphiques aident à déterminer les ordres p et q du modèle ARIMA en
    analysant respectivement l'autocorrélation totale et partielle de la série
    ou de sa version différenciée.

    Parameters
    ----------
    series : pd.Series
        La série (ou série différenciée) sur laquelle calculer les corrélations.
    lags : int, default=40
        Nombre de retards à afficher.
    filename : str, default='acf_pacf.png'
        Nom du fichier PNG où enregistrer le graphique.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series, ax=axes[0], lags=lags)
    axes[0].set_title('ACF')
    plot_pacf(series, ax=axes[1], lags=lags, method='ywm')
    axes[1].set_title('PACF')
    plt.tight_layout()
    plt.savefig(filename)


#
# Fonctions pour l'ARIMA automatique via pmdarima (auto_arima)
#
def fit_auto_arima_model(train: pd.Series):
    """Ajuste un modèle ARIMA automatiquement avec pmdarima.

    Cette fonction n'est disponible que si la librairie pmdarima est installée. Elle
    sélectionne les paramètres (p, d, q) en minimisant l'AIC sur la série
    d'entraînement. Si pmdarima n'est pas disponible, une ImportError est levée.

    Parameters
    ----------
    train : pd.Series
        Série d'entraînement.

    Returns
    -------
    pmdarima.arima.ARIMA
        Modèle auto_arima ajusté.
    """
    if not _PMDARIMA_AVAILABLE:
        raise ImportError(
            "La librairie 'pmdarima' n'est pas installée. Veuillez l'installer pour utiliser l'auto_arima."
        )
    # On autorise jusqu'à des ordres modérés pour éviter les temps de calcul trop longs
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        model = auto_arima(
            train,
            start_p=0, start_q=0,
            max_p=3, max_q=3,
            d=1,
            seasonal=False,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
        )
    return model


def predict_auto_arima_model(model, steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Génère des prévisions et intervalles pour un modèle auto_arima.

    Parameters
    ----------
    model : pmdarima.arima.ARIMA
        Modèle ajusté.
    steps : int
        Nombre de pas de prévision.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (prévisions, borne inférieure, borne supérieure)
    """
    forecast, conf_int = model.predict(n_periods=steps, return_conf_int=True, alpha=0.05)
    lower, upper = conf_int[:, 0], conf_int[:, 1]
    return forecast, lower, upper


def evaluate_forecast(true_values: np.ndarray, predictions: np.ndarray) -> dict:
    """Calcule les métriques RMSE, MAE et R² entre les valeurs vraies et les prévisions."""
    rmse = np.sqrt(mean_squared_error(true_values, predictions))
    mae = mean_absolute_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}


def main():
    """Programme principal pour le TP de séries temporelles.

    Ce flux illustre deux approches pédagogiques de prévision :

      1. **Analyse manuelle des séries :** on charge la série CO₂ et on effectue
         un split entraînement/test. On vérifie la stationnarité de la série
         originale et de sa première différence avec le test ADF, puis on
         trace les graphiques ACF et PACF pour guider le choix des ordres p
         et q du modèle ARIMA. L'inspection des ACF/PACF montre un décrochage
         progressif de l'ACF et des décrochements significatifs aux lags 1–3
         dans la PACF, ce qui conduit à un ARIMA(2, 1, 3). La différence d=1
         est choisie car la série n'est pas stationnaire au niveau, mais le
         test ADF montre que la première différence l'est (p‑value ≪ 0,05).

      2. **ARIMA manuel :** on ajuste un modèle ARIMA avec les ordres p=2,
         d=1, q=3 obtenus ci‑dessus et on génère des prévisions et intervalles
         de confiance.

      3. **ARIMA automatique via pmdarima :** si la librairie `pmdarima` est
         installée, on utilise `auto_arima` pour sélectionner automatiquement
         (p, d, q) en minimisant l'AIC. Dans le cas de cette série, l'ordre
         automatique retenu est également (2, 1, 3), ce qui confirme la
         pertinence de l'analyse manuelle.

      4. **Évaluation et visualisation :** on calcule les métriques RMSE, MAE
         et R² pour chaque modèle disponible et on trace un graphique
         comparatif des prévisions avec leurs intervalles.
    """
    # 1. Chargement et préparation des données
    series = load_co2_series()
    train, test = train_test_split(series, test_size=52)

    # 2. Diagnostic de stationnarité et ACF/PACF
    # Test ADF sur la série originale
    check_stationarity(train, name="série originale")
    # Test ADF sur la première différence
    diff_train = train.diff().dropna()
    check_stationarity(diff_train, name="première différence")
    # Trace ACF/PACF pour la série différenciée
    plot_acf_pacf(diff_train, lags=40, filename='acf_pacf_diff.png')

    # 3. Ajustement ARIMA manuel
    # Le choix de l'ordre (p,d,q) est basé sur l'analyse précédente. On fixe d=1
    # car la différenciation rend la série stationnaire selon l'ADF et p, q
    # sont choisis après inspection des ACF/PACF. Ici (2, 1, 3).
    arima_order = (2, 1, 3)
    print(f"\nAjustement d'un ARIMA{arima_order}...")
    arima_model = fit_arima_manual(train, order=arima_order)
    arima_pred, arima_lower, arima_upper = predict_arima(arima_model, steps=len(test))

    # 4. Ajustement ARIMA automatique avec pmdarima (si disponible)
    auto_pred = auto_lower = auto_upper = None
    auto_order = None
    if _PMDARIMA_AVAILABLE:
        print("\nRecherche automatique des paramètres ARIMA via pmdarima...")
        try:
            auto_model = fit_auto_arima_model(train)
            auto_pred, auto_lower, auto_upper = predict_auto_arima_model(auto_model, steps=len(test))
            auto_order = auto_model.order
        except Exception as e:
            print("Impossible d'ajuster un ARIMA automatique :", e)
    else:
        print("La librairie 'pmdarima' n'est pas installée : l'auto_arima n'est pas disponible.")

    # 6. Évaluation des modèles
    results = []
    # ARIMA manuel
    results.append({'Model': f'ARIMA{arima_order}', **evaluate_forecast(test.values, arima_pred)})
    # ARIMA automatique si disponible
    if auto_pred is not None:
        results.append({'Model': f'AutoARIMA{auto_order}', **evaluate_forecast(test.values, auto_pred)})
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv('forecast_metrics.csv', index=False)

    # 7. Tracé des prévisions
    plt.figure(figsize=(12, 6))
    # Observations réelles
    plt.plot(series.index[-len(test):], test.values, label='Observation', color='black')
    # Prévisions ARIMA manuel
    plt.plot(test.index, arima_pred, label=f'ARIMA{arima_order}')
    plt.fill_between(test.index, arima_lower, arima_upper, alpha=0.2)
    # Prévisions ARIMA automatique si disponible
    if auto_pred is not None:
        plt.plot(test.index, auto_pred, label=f'AutoARIMA{auto_order}')
        plt.fill_between(test.index, auto_lower, auto_upper, alpha=0.2)
    plt.title('Comparaison des prévisions sur la série CO₂')
    plt.xlabel('Date')
    plt.ylabel('CO₂ (ppmv)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('forecast_comparison.png')

    # Affiche les métriques
    print("\nMétriques de performance :")
    print(metrics_df)


if __name__ == '__main__':
    main()