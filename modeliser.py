"""Prédiction du risque d'hospitalisation ou de passage aux urgences à 12 mois
sur une cohorte Synthea : jeu de test mis de côté, validation croisée,
discrimination, calibration, importance des variables et équité selon le sexe
et le revenu."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

d = pd.read_csv("cohorte synthea.csv")
d["log_revenu"] = np.log1p(d["revenu"])
y = d["hospit_ou_urgences_12mois"]
num = ["age", "imc", "pas", "nb_pathologies", "nb_medicaments", "hospit_12mois_avant",
       "urgences_12mois_avant", "consult_12mois_avant", "log_revenu"]
bin_ = ["tabac_actuel_ou_ancien", "diabete", "hypertension", "insuffisance_cardiaque", "bpco",
        "insuffisance_renale", "asthme", "depression"]
cat = ["sexe"]
X = d[num + bin_ + cat]

# Jeu de test (25 %) mis de côté avant toute modélisation (une ligne par patient)
idx_app, idx_test = train_test_split(d.index, test_size=0.25, stratify=y, random_state=2026)
app = d.index.isin(idx_app); test = d.index.isin(idx_test)

prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("std", StandardScaler())]), num),
    ("bin", "passthrough", bin_),
    ("cat", OneHotEncoder(drop="first"), cat)])
modeles = {
    "Régression logistique": Pipeline([("prep", prep), ("m", LogisticRegression(max_iter=2000))]),
    "Gradient boosting": Pipeline([("prep", prep), ("m", HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0,
        early_stopping=True, random_state=2026))]),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
res, proba = {}, {}
for nom, pipe in modeles.items():
    s = cross_validate(pipe, X[app], y[app], cv=cv,
                       scoring={"auc": "roc_auc", "brier": "neg_brier_score"})
    pipe.fit(X[app], y[app])
    p = pipe.predict_proba(X[test])[:, 1]; proba[nom] = p
    yt = y[test].to_numpy()
    rng = np.random.default_rng(1); boots = []
    for _ in range(1000):
        i = rng.integers(0, len(yt), len(yt))
        if len(set(yt[i])) == 2: boots.append(roc_auc_score(yt[i], p[i]))
    res[nom] = dict(auc_cv=round(s["test_auc"].mean(), 3), auc_cv_et=round(s["test_auc"].std(), 3),
                    auc_test=round(roc_auc_score(yt, p), 3),
                    auc_test_ic=[round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)],
                    brier_test=round(brier_score_loss(yt, p), 4))

yt = y[test].to_numpy()
meilleur = max(res, key=lambda k: res[k]["auc_test"])
pm = proba[meilleur]
# Équité : AUC et calibration moyenne par sous-groupe
dt = d[test].copy(); dt["p"] = pm
dt["quintile_revenu"] = pd.qcut(dt["revenu"], 5, labels=["Q1 (plus faible)", "Q2", "Q3", "Q4", "Q5 (plus élevé)"])
equite = []
for var in ["sexe", "quintile_revenu"]:
    for g, sub in dt.groupby(var, observed=True):
        if sub[y.name].nunique() == 2:
            equite.append(dict(variable=var, groupe=str(g), n=len(sub),
                               taux_observe=round(sub[y.name].mean(), 3),
                               risque_predit_moyen=round(sub["p"].mean(), 3),
                               auc=round(roc_auc_score(sub[y.name], sub["p"]), 3)))
sortie = dict(
    cohorte=dict(adultes=len(d), taux_evenement=round(y.mean(), 3),
                 apprentissage=int(app.sum()), test=int(test.sum())),
    performances=res,
    reference_brier=round(brier_score_loss(yt, np.full(len(yt), y[app].mean())), 4),
    modele_retenu=meilleur, equite=equite)
json.dump(sortie, open("performances.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(sortie, ensure_ascii=False, indent=2))

coul = {"Régression logistique": "#2b5d63", "Gradient boosting": "#c0583e"}
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for nom, p in proba.items():
    fpr, tpr, _ = roc_curve(yt, p)
    ax[0].plot(fpr, tpr, color=coul[nom], lw=2, label=f"{nom} (AUC {res[nom]['auc_test']:.2f})")
    frac, moy = calibration_curve(yt, p, n_bins=10, strategy="quantile")
    ax[1].plot(moy, frac, "o-", color=coul[nom], lw=2, label=nom)
ax[0].plot([0, 1], [0, 1], "--", color="grey")
ax[0].set(xlabel="1 - spécificité", ylabel="Sensibilité", title="Discrimination (jeu de test)")
lim = max(ax[1].get_xlim()[1], ax[1].get_ylim()[1]); ax[1].plot([0, lim], [0, lim], "--", color="grey")
ax[1].set(xlabel="Risque prédit", ylabel="Fréquence observée", title="Calibration (déciles de risque)")
for a in ax: a.legend(frameon=False, fontsize=9); a.spines[["top", "right"]].set_visible(False)
fig.suptitle("Synthea : hospitalisation ou urgences à 12 mois", x=0.02, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig("1 discrimination et calibration.png", dpi=170)

pi = permutation_importance(modeles[meilleur], X[test], y[test], scoring="roc_auc",
                            n_repeats=20, random_state=2026)
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values()
fig, ax = plt.subplots(figsize=(7, 4.8))
ax.barh(imp.index, imp.values, color="#3f7f86")
ax.set(xlabel="Perte d'AUC quand la variable est permutée", title=f"Importance des variables ({meilleur.lower()})")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("2 importance des variables.png", dpi=170)

eq = pd.DataFrame(equite)
fig, ax = plt.subplots(figsize=(8, 4.4))
yy = np.arange(len(eq))[::-1]
ax.scatter(eq["taux_observe"], yy, color="#2b5d63", label="Taux observé", zorder=3)
ax.scatter(eq["risque_predit_moyen"], yy, color="#c0583e", marker="D", label="Risque prédit moyen", zorder=3)
for yi, (a, b) in zip(yy, zip(eq["taux_observe"], eq["risque_predit_moyen"])):
    ax.plot([a, b], [yi, yi], color="grey", lw=1)
ax.set_yticks(yy); ax.set_yticklabels([f"{v} · {g} (AUC {a:.2f})" for v, g, a in zip(eq["variable"].map({"sexe": "Sexe", "quintile_revenu": "Revenu"}), eq["groupe"], eq["auc"])], fontsize=8.5)
ax.set_xlabel("Hospitalisation ou urgences à 12 mois"); ax.legend(frameon=False, fontsize=9)
ax.set_title("Équité : calibration et discrimination par sous-groupe (jeu de test)", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("3 equite par sous-groupe.png", dpi=170)
