# Customer Churn Analysis

Warum kündigen Kunden und lässt sich das vorhersagen, bevor es passiert?

Diese Frage stand am Anfang des Projekts. Das Ergebnis ist eine vollständige Analyse des Telco Customer Churn Datasets, drei trainierte Machine-Learning-Modelle und eine interaktive Streamlit-App, mit der sich das Abwanderungsrisiko einzelner Kunden in Echtzeit berechnen lässt.

---

## Hintergrund

Das zugrunde liegende Dataset stammt von Kaggle und wurde ursprünglich von IBM Watson Analytics bereitgestellt. Es enthält Daten von **7.043 Kunden** eines US-amerikanischen Telekommunikationsanbieters mit demografischen Informationen, genutzten Services, Vertragslaufzeiten und der entscheidenden Variable: Hat der Kunde gekündigt oder nicht?

27% der Kunden im Dataset haben gekündigt. Das klingt nach einer überschaubaren Minderheit, ist aber aus Unternehmenssicht ein erheblicher Verlust. Neukundengewinnung ist typischerweise fünf bis sieben Mal teurer als Bestandskundenpflege.

---

## Was steckt in diesem Repo?

Das Projekt ist in drei Teile gegliedert, die aufeinander aufbauen.

**Explorative Datenanalyse** (`notebooks/01_EDA.ipynb`)
Hier geht es darum, das Dataset wirklich zu verstehen. Welche Variablen hängen mit Churn zusammen? Wo sind die stärksten Signale? Das Notebook enthält alle Visualisierungen und die ersten Business-Insights.

**Modellentwicklung** (`notebooks/02_modeling.ipynb`)
Drei Modelle werden trainiert und direkt miteinander verglichen: Logistic Regression, Random Forest und XGBoost. Am Ende gibt es einen klaren Gewinner mit einer ausführlichen Begründung.

**Interaktive Streamlit-App** (`app.py`)
Das trainierte Modell ist in eine Web-App eingebettet. Wer wissen möchte, ob ein konkreter Kunde gefährdet ist, gibt einfach die relevanten Parameter ein und bekommt sofort einen Risiko-Score zurück.

---

## Ergebnisse auf einen Blick

Fünf Faktoren erklären den Großteil des Churns.

**Vertragstyp** ist der mit Abstand stärkste Faktor. Kunden mit Monatsvertrag kündigen zu 43%, bei Jahresverträgen sind es 11% und bei Zweijahresverträgen nur noch 3%. Wer langfristig bindet, verliert kaum Kunden.

**Monatliche Kosten** spielen ebenfalls eine Rolle. Churner zahlen im Median 80 USD pro Monat, loyale Kunden 65 USD. Die Kombination aus hohen Kosten und fehlendem Mehrwert ist besonders riskant.

**Die ersten 12 Monate** sind die kritischste Phase. Danach sinkt die Abwanderungsrate deutlich. Kunden die zwei Jahre oder länger bleiben kündigen praktisch kaum noch.

**Schutz und Support-Services** machen einen erheblichen Unterschied. Ohne OnlineSecurity oder TechSupport liegt die Churn-Rate bei über 41%, mit diesen Services bei nur 14 bis 15%. Das ist der größte Retention-Hebel im gesamten Dataset.

**Zahlungsmethode** ist überraschend aussagekräftig. Kunden die per Electronic Check zahlen kündigen zu 45,3%, fast dreimal häufiger als Kunden mit automatischen Zahlungsmethoden.

---

## Modellvergleich

Drei Modelle wurden trainiert, alle auf denselben Daten (80/20 Split mit StandardScaling):

| Modell | AUC-ROC | Accuracy | Precision | Recall |
|---|---|---|---|---|
| **Logistic Regression** | **0.840** | **80%** | **64%** | **55%** |
| Random Forest | 0.825 | 80% | | |
| XGBoost | 0.823 | 78% | | |

Das finale Modell ist **Logistic Regression**, nicht trotz seiner Einfachheit sondern wegen ihr. Bei rund 7.000 Zeilen und vielen kategorischen Features neigen komplexere Modelle dazu, auf Trainingsartefakte zu fitten. Logistic Regression generalisiert hier besser, ist vollständig interpretierbar und liefert direkt Wahrscheinlichkeitswerte statt nur binäre Vorhersagen.

---

## Die Streamlit-App

Das Modell ist in eine interaktive Web-App eingebettet, die sich lokal starten lässt:

```bash
streamlit run app.py
```

In der App gibt man für einen einzelnen Kunden folgende Parameter ein:

- Vertragstyp (Monat, Jahr oder 2 Jahre)
- Vertragslaufzeit in Monaten
- Monatliche Kosten
- Zahlungsmethode
- Internet-Service-Typ
- OnlineSecurity und TechSupport (ja oder nein)

Das Modell berechnet daraus eine **Churn-Wahrscheinlichkeit in Prozent**, die farbcodiert angezeigt wird. Rot bei hohem Risiko, grün bei niedrigem. Zusätzlich werden die wichtigsten Risikofaktoren für den jeweiligen Kunden erklärt.

Das macht die App besonders nützlich für Retention-Teams, die nicht mit rohen Modelloutputs arbeiten wollen, sondern schnelle und verständliche Einschätzungen brauchen.

Unter diesen Link gelangt man direkt zur App:
https://churn-analysis-buck.streamlit.app

---

## Business-Empfehlungen

Die Analyse zeigt, dass gezieltes Handeln auf wenigen Hebeln schon viel bewirken kann.

**Jahresverträge aktiv bewerben.** Ein Rabatt oder ein kleines Incentive kann einen Monatskunden zu einem Jahreskunden machen und die Churn-Wahrscheinlichkeit fast vierteln.

**Strukturiertes Onboarding in den ersten 90 Tagen.** Proaktiver Kundenkontakt nach 30, 60 und 90 Tagen würde Early Churn deutlich reduzieren. Der erste Monat entscheidet oft.

**OnlineSecurity und TechSupport als Bundle vermarkten.** Diese Services halbieren die Churn-Rate, was ein starkes Argument für aktives Upselling direkt beim Vertragsabschluss ist.

**Electronic-Check-Kunden als Risikogruppe markieren.** Ein gezielter Anreiz zum Wechsel auf automatische Zahlung ist eine einfache Maßnahme mit messbarer Wirkung.

---

## Projektstruktur

```
churn-analysis/
│
├── app.py                        # Streamlit-App (Echtzeit-Risikovorhersage)
│
├── data/
│   └── raw/
│       └── Telco_Customer_Churn.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb              # Explorative Datenanalyse
│   └── 02_modeling.ipynb         # Modelltraining und Vergleich
│
├── models/                       # Gespeicherte Modelle (.pkl)
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   └── scaler.pkl
│
├── reports/
│   └── figures/                  # Alle generierten Visualisierungen
│
├── requirements.txt
└── README.md
```

---

## Installation und Start

```bash
git clone https://github.com/Buck-Data/customer-churn-analysis.git
cd churn-analysis

python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

Notebooks starten:
```bash
jupyter notebook
```

Streamlit-App starten:
```bash
streamlit run app.py
```

---

## Tech Stack

Python 3.11, pandas, numpy, scikit-learn, XGBoost, matplotlib, seaborn, Streamlit, Jupyter

---

## Dataset

Telco Customer Churn Dataset von Kaggle, bereitgestellt von IBM Watson Analytics. Das Dataset enthält demografische Daten, Vertragsinformationen, genutzte Services und die Churn-Variable als Zielvariable.
