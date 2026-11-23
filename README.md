# Risque d'hospitalisation ou de passage aux urgences à 12 mois (données Synthea)

Ce projet prédit, pour des adultes suivis en soins primaires, le risque d'au moins une hospitalisation ou d'un passage aux urgences dans l'année, à partir de leur dossier médical. Il compare une régression logistique et un modèle de gradient boosting, et vérifie si le modèle est aussi juste pour tous les groupes (sexe, revenu).

Auteur : Armel Mahutin KANHONOU · [Portfolio](https://armel-mahutin.github.io) · ORCID [0009-0005-1764-1327](https://orcid.org/0009-0005-1764-1327)

## Données

[Synthea](https://github.com/synthetichealth/synthea) (MITRE) est un générateur open source de dossiers médicaux **synthétiques** mais réalistes : naissances, consultations, maladies chroniques, traitements, examens et hospitalisations, simulés à partir de statistiques de santé publiques. Aucune personne réelle n'y figure, ce qui permet de publier les données.

- Population générée : 4 596 patients du Massachusetts, 10 ans d'historique (graine 2026).
- Cohorte analysée : **3 058 adultes** vivants et vus au moins une fois dans les 2 ans précédant la date de référence (23/11/2025).
- Issue : au moins une hospitalisation ou un passage aux urgences dans les 12 mois suivants : **16,6 %**.

## Variables (toutes mesurées avant la date de référence)

Âge, sexe, revenu, IMC, pression artérielle systolique, tabagisme actuel ou ancien, nombre de maladies chroniques actives, diabète, hypertension, insuffisance cardiaque, BPCO, insuffisance rénale chronique, asthme, dépression, nombre de médicaments en cours, hospitalisations, passages aux urgences et consultations des 12 mois précédents.

Aucune information postérieure à la date de référence n'entre dans les variables, ce qui évite la fuite de données dans le temps.

## Démarche

1. Jeu de test de 25 % mis de côté avant toute modélisation (une ligne par patient).
2. Prétraitement dans un pipeline : imputation, standardisation, encodage, appris sur les seules données d'apprentissage.
3. Validation croisée stratifiée à 5 plis, puis évaluation sur le jeu de test : AUC avec IC à 95 % (bootstrap), score de Brier, calibration.
4. Importance des variables par permutation.
5. **Équité** : taux observé, risque prédit moyen et AUC par sexe et par quintile de revenu.

## Résultats

| Modèle | AUC en validation croisée | AUC sur le jeu de test (IC 95 %) | Score de Brier |
|---|---|---|---|
| Régression logistique | 0,646 (± 0,039) | 0,585 (0,524 – 0,643) | 0,133 |
| Gradient boosting | 0,650 (± 0,011) | **0,659** (0,606 – 0,714) | **0,130** |
| Référence : taux moyen | – | 0,500 | 0,139 |

![Discrimination et calibration](1%20discrimination%20et%20calibration.png)

![Importance des variables](2%20importance%20des%20variables.png)

![Équité par sous-groupe](3%20equite%20par%20sous-groupe.png)

## Lecture

- **La prédiction est modeste (AUC ≈ 0,66).** C'est un résultat honnête : le recours aux urgences dépend aussi d'événements aigus imprévisibles à partir du seul dossier médical. Le modèle n'améliore que légèrement le score de Brier par rapport au taux moyen.
- **Le gradient boosting fait un peu mieux** que la régression logistique sur le jeu de test, mais les intervalles de confiance se chevauchent.
- **Équité** : le risque prédit s'écarte du risque observé en sens opposés chez les femmes (19,4 % prédit contre 16,7 % observé, IC 95 % exact 13,2–20,7) et chez les hommes (13,4 % contre 16,5 %, IC 95 % 12,9–20,8). Les deux valeurs prédites restent dans l'intervalle de confiance du taux observé : avec environ 400 personnes par sexe dans le jeu de test, ces écarts peuvent relever du hasard. Pour conclure, il faudrait une cohorte plus grande (Synthea permet d'en générer une gratuitement), puis recalibrer par sexe si l'écart se confirme. Les écarts entre quintiles de revenu reposent sur environ 150 personnes par groupe et sont à interpréter avec prudence.

## Limites

- Données synthétiques : les relations reflètent les règles du simulateur, pas forcément la réalité. Le projet sert à montrer la démarche, pas à produire un outil clinique.
- Le tabagisme actif est très rare dans cette population simulée, d'où le regroupement « actuel ou ancien ».
- Pas de validation externe.

## Reproduire l'analyse

1. Générer la population avec Synthea (Java 17), version `5441eef` :

```bash
java -jar synthea-with-dependencies.jar -s 2026 -r 20261123 -p 4000 \
  --exporter.csv.export=true --exporter.fhir.export=false \
  --exporter.years_of_history=10 Massachusetts
```

2. Construire la cohorte, puis modéliser :

```bash
pip install -r requirements.txt
python "construire cohorte.py" output/csv
python modeliser.py
```

La cohorte obtenue (`cohorte synthea.csv`) est fournie pour lancer directement `modeliser.py`.

## Fichiers

```
construire cohorte.py    cohorte et variables à partir des CSV de Synthea
modeliser.py             modèles, validation, calibration, équité, figures
cohorte synthea.csv      cohorte analysée (3 058 adultes)
performances.json        résultats détaillés
*.png                    figures
```
