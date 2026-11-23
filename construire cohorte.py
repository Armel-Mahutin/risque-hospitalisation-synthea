"""Construit, à partir des fichiers CSV de Synthea, une cohorte d'adultes
suivis à une date de référence (un an avant la fin des données), avec :
- des variables connues AVANT cette date (âge, sexe, revenu, pathologies
  chroniques, traitements, recours aux soins, IMC, pression artérielle, tabac) ;
- l'issue : au moins une hospitalisation ou un passage aux urgences
  dans les 12 mois qui suivent."""
import sys
import numpy as np
import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "synthea/csv"
to_dt = lambda s: pd.to_datetime(s, utc=True, errors="coerce").dt.tz_localize(None)

pat = pd.read_csv(f"{src}/patients.csv")
pat["BIRTHDATE"] = pd.to_datetime(pat["BIRTHDATE"]); pat["DEATHDATE"] = pd.to_datetime(pat["DEATHDATE"])
enc = pd.read_csv(f"{src}/encounters.csv", usecols=["START", "PATIENT", "ENCOUNTERCLASS"])
enc["START"] = to_dt(enc["START"])
fin = enc["START"].max().normalize()
ref = fin - pd.Timedelta(days=365)

# Population : adultes vivants à la date de référence, vus au moins une fois dans les 2 ans avant
vus = enc[(enc.START < ref) & (enc.START >= ref - pd.Timedelta(days=730))].PATIENT.unique()
p = pat[pat.Id.isin(vus) & (pat.DEATHDATE.isna() | (pat.DEATHDATE > ref))].copy()
p["age"] = ((ref - p.BIRTHDATE).dt.days / 365.25).round(1)
p = p[p.age >= 18].copy()

# Issue : hospitalisation ou urgences dans les 12 mois après la date de référence
apres = enc[(enc.START > ref) & (enc.START <= fin) & enc.ENCOUNTERCLASS.isin(["inpatient", "emergency"])]
p["hospit_ou_urgences_12mois"] = p.Id.isin(apres.PATIENT).astype(int)

# Recours aux soins dans les 12 mois AVANT
avant = enc[(enc.START < ref) & (enc.START >= ref - pd.Timedelta(days=365))]
for nom, classe in {"hospit_12mois_avant": "inpatient", "urgences_12mois_avant": "emergency",
                    "consult_12mois_avant": "ambulatory"}.items():
    p[nom] = p.Id.map(avant[avant.ENCOUNTERCLASS == classe].groupby("PATIENT").size()).fillna(0)

# Pathologies chroniques actives à la date de référence
cond = pd.read_csv(f"{src}/conditions.csv", usecols=["START", "STOP", "PATIENT", "DESCRIPTION"])
cond["START"] = to_dt(cond["START"]); cond["STOP"] = to_dt(cond["STOP"])
act = cond[(cond.START <= ref) & (cond.STOP.isna() | (cond.STOP > ref))]
act = act[~act.DESCRIPTION.str.contains(r"\((?:finding|situation)\)", regex=True)]  # garder les maladies
p["nb_pathologies"] = p.Id.map(act.groupby("PATIENT").size()).fillna(0)
desc = act.DESCRIPTION.str.lower()
for nom, motif in {"diabete": "diabetes", "hypertension": "hypertension",
                   "insuffisance_cardiaque": "heart failure", "bpco": "chronic obstructive",
                   "insuffisance_renale": "chronic kidney disease", "asthme": "asthma",
                   "depression": "depressive"}.items():
    p[nom] = p.Id.isin(set(act.loc[desc.str.contains(motif, regex=False), "PATIENT"])).astype(int)

# Médicaments actifs à la date de référence
med = pd.read_csv(f"{src}/medications.csv", usecols=["START", "STOP", "PATIENT"])
med["START"] = to_dt(med["START"]); med["STOP"] = to_dt(med["STOP"])
ma = med[(med.START <= ref) & (med.STOP.isna() | (med.STOP > ref))]
p["nb_medicaments"] = p.Id.map(ma.groupby("PATIENT").size()).fillna(0)

# Dernières mesures avant la date de référence (lecture par morceaux)
cibles = {"Body mass index (BMI) [Ratio]": "imc", "Systolic Blood Pressure": "pas",
          "Tobacco smoking status": "tabac"}
morceaux = []
for ch in pd.read_csv(f"{src}/observations.csv", usecols=["DATE", "PATIENT", "DESCRIPTION", "VALUE"],
                      chunksize=1_000_000):
    ch = ch[ch.DESCRIPTION.isin(cibles) & ch.PATIENT.isin(p.Id)]
    if len(ch): morceaux.append(ch)
obs = pd.concat(morceaux); obs["DATE"] = to_dt(obs["DATE"])
obs = obs[obs.DATE < ref].sort_values("DATE").groupby(["PATIENT", "DESCRIPTION"]).tail(1)
for desc_, nom in cibles.items():
    o = obs[obs.DESCRIPTION == desc_].set_index("PATIENT")["VALUE"]
    p[nom] = p.Id.map(o)
p["imc"] = pd.to_numeric(p["imc"], errors="coerce"); p["pas"] = pd.to_numeric(p["pas"], errors="coerce")
p["tabac_actuel_ou_ancien"] = p["tabac"].fillna("").str.contains("Smokes|Ex-smoker", case=False).astype(int)

p["sexe"] = p["GENDER"]; p["revenu"] = p["INCOME"]
cols = ["Id", "age", "sexe", "revenu", "imc", "pas", "tabac_actuel_ou_ancien", "nb_pathologies", "diabete",
        "hypertension", "insuffisance_cardiaque", "bpco", "insuffisance_renale", "asthme",
        "depression", "nb_medicaments", "hospit_12mois_avant", "urgences_12mois_avant",
        "consult_12mois_avant", "hospit_ou_urgences_12mois"]
out = p[cols].rename(columns={"Id": "patient"})
out.to_csv("cohorte synthea.csv", index=False)
print(f"Date de référence : {ref.date()} ; {len(out)} adultes ; "
      f"hospitalisation ou urgences à 12 mois : {out.hospit_ou_urgences_12mois.mean():.1%}")
print(out.isna().mean().round(3)[out.isna().mean() > 0])
