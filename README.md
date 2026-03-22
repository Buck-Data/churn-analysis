# Customer Churn Analysis

## Was ist das Projekt?

Dieses Projekt analysiert, warum Kunden eines Telekommunikationsunternehmens kündigen. Ziel war es, aus echten Kundendaten herauszufinden welche Faktoren am stärksten mit Churn zusammenhängen und darauf basierend ein Modell zu bauen, das gefährdete Kunden frühzeitig erkennt.

Die Analyse basiert auf dem Telco Customer Churn Dataset von Kaggle mit 7043 Kunden und 21 Features.

---

## Ergebnisse auf einen Blick

Nach der Analyse haben sich fünf klare Muster herauskristallisiert:

Kunden mit Monatskontrakten kündigen mit einer Rate von ca. 43%, während Jahreskunden nur bei 11% liegen. Das ist der stärkste einzelne Faktor im gesamten Dataset.

Churner zahlen im Median 80 USD pro Monat, verglichen mit 65 USD bei treuen Kunden. Günstigere Kunden bleiben deutlich loyaler.

Die ersten 12 Monate sind die gefährlichste Phase. Danach sinkt die Abwanderungsrate kontinuierlich. Kunden die länger als zwei Jahre bleiben kündigen kaum noch.

Kunden ohne OnlineSecurity oder TechSupport churnen mit über 41%, während Kunden mit diesen Services nur bei 14 bis 15% liegen. Das ist der stärkste Hebel für Retention.

Kunden die per Electronic Check zahlen kündigen zu 45.3%, also fast dreimal häufiger als Kunden mit automatischen Zahlungsmethoden wie Kreditkarte oder Lastschrift.

---

## Modell Performance

Ich habe drei Modelle trainiert und miteinander verglichen:

| Modell | AUC-ROC | Accuracy |
|--------|---------|----------|
| Logistic Regression | 0.840 | 80% |
| Random Forest | 0.825 | 80% |
| XGBoost | 0.823 | 78% |

Das finale Modell ist Logistic Regression mit einem AUC-ROC Score von 0.84. Trotz einfacherer Architektur hat es die komplexeren Modelle geschlagen, was bei diesem Dataset mit vielen kategorischen Features und ca. 7000 Zeilen nicht ungewöhnlich ist.

---

## Business Empfehlungen

Basierend auf den Findings würde ich folgende Maßnahmen vorschlagen:

Monatskunden sollten aktiv zu Jahresverträgen konvertiert werden, zum Beispiel durch Rabatte oder exklusive Vorteile bei längerer Bindung.

Ein strukturiertes Onboarding in den ersten 90 Tagen würde Early Churn deutlich reduzieren. Konkret: proaktiver Kundenkontakt nach 30, 60 und 90 Tagen.

OnlineSecurity und TechSupport sollten als Bundle aktiv vermarktet werden, da sie die Churn Rate fast dritteln.

Electronic Check Kunden sollten als Risikogruppe markiert werden. Ein Anreiz zum Wechsel auf automatische Zahlung wäre eine einfache und günstige Maßnahme.

---

## Projektstruktur

```
customer-churn-analysis/
│
├── data/
│   ├── raw/                  
│   └── processed/            
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_modeling.ipynb
│
├── reports/
│   └── figures/              
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/Buck-Data/customer-churn-analysis.git
cd customer-churn-analysis

python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

---

## Tech Stack

Python 3.11, pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost

---

## Dataset

Telco Customer Churn Dataset von Kaggle, bereitgestellt von IBM Watson Analytics. Das Dataset enthält demografische Daten, Vertragsinformationen, genutzte Services und die Churn-Variable als Zielvariable.