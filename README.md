# RINL Delay Analysis & Predictive Maintenance System

## Project Title

**AI-Based Delay Analysis and Predictive Maintenance System for RINL Visakhapatnam Steel Plant**

This is a simple, professional, ready-to-run Python Streamlit application developed for delay analysis, shop-wise monitoring, equipment-wise delay tracking, agency-wise delay analysis, conveyor delay analysis, monsoon delay study, and predictive maintenance risk identification.

The project is designed as a prototype suitable for a B.Tech CSE internship demonstration at **RINL Visakhapatnam Steel Plant**.

---

## Project Objective

The main objective of this application is to analyze industrial delay data and identify delay patterns across shops, equipment, agencies, conveyors, seasons, and delay descriptions.

The system also provides a basic predictive maintenance module that estimates equipment risk levels using delay duration, delay frequency, cumulative delay, conveyor involvement, and monsoon impact.

---

## Files Included

The application uses a very simple project structure:

```text
rinl_delay_analysis_system/
│
├── app.py
├── requirements.txt
├── sample_delays_data.csv
└── README.md
```

No complex backend, database, Docker, or separate frontend is required.

---

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn
- Joblib
- OpenPyXL

---

## Dataset

The application loads the CSV file directly from the same folder:

```text
sample_delays_data.csv
```

Expected columns include:

```text
DEL_DATE, SHOP_CODE, MATERIAL, RAKE, DELAY_FROM, DELAY_TO,
DELAY_DURN, CUM_DELAY, EQPT, SUB_EQPT, REMARKS,
DELAY_DET_CODE, AGENCY_CODE, DELAY_FREQ, CONTINUED,
EXPECTED_DOC, EFF_DURATION, DELAY_ID
```

The application is designed to handle missing columns safely and show warnings instead of crashing.

---

## Main Features

### 1. Executive Dashboard

Shows key performance indicators such as:

- Total delay records
- Total delay duration
- Average delay duration
- Most affected shop
- Most affected equipment
- Most responsible agency
- Conveyor delay count
- High-risk equipment count

Also includes interactive charts for monthly trend, top shops, top equipment, top agencies, and delay category distribution.

---

### 2. Shop-wise Analysis

Provides shop-wise delay duration, delay count, average delay, and monthly shop delay trend.

---

### 3. Equipment-wise Analysis

Provides equipment-wise and sub-equipment-wise delay analysis, repeated equipment issues, and equipment risk table.

---

### 4. Agency-wise Analysis

Shows agency-wise total delay, delay count, average delay, and agency ranking.

---

### 5. Duration-wise Analysis

Classifies delays into:

- Minor Delay: 0 to 30 minutes
- Moderate Delay: 31 to 120 minutes
- Major Delay: 121 to 360 minutes
- Critical Delay: above 360 minutes

---

### 6. Conveyor-wise Analysis

Identifies conveyor-related delays using keywords from equipment, sub-equipment, and remarks.

Keywords used:

```text
CONVEYOR, CONV, BELT, BCN, TCN, CC
```

---

### 7. Delay Description Analysis

Classifies delay reasons into:

- Mechanical
- Electrical
- Conveyor
- Material Handling
- Operational
- Agency Related
- Other

---

### 8. Season / Monsoon Analysis

Classifies delays by season:

- Summer: March, April, May
- Monsoon: June, July, August, September
- Post-Monsoon: October, November
- Winter: December, January, February

This helps identify monsoon-related conveyor and equipment delay risks.

---

### 9. Predictive Maintenance

Uses a practical risk scoring method and a simple Random Forest Classifier where possible.

Risk levels:

- Low Risk
- Medium Risk
- High Risk
- Critical Risk

The prediction module provides:

- Shop code
- Equipment
- Sub-equipment
- Delay frequency
- Total delay duration
- Risk level
- Maintenance recommendation

---

### 10. Download Report

Allows the user to download filtered summary data as a CSV report.

---

## How to Run the Project

### Step 1: Open the project folder

```bash
cd rinl_delay_analysis_system
```

### Step 2: Install required libraries

```bash
pip install -r requirements.txt
```

### Step 3: Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## Notes

- Keep `sample_delays_data.csv` in the same folder as `app.py`.
- Do not rename the CSV file unless you also update the filename inside `app.py`.
- The application uses only Python-based tools and is designed for simple execution.
- The dashboard has a professional industrial style using custom CSS.

---

## Suitable For

- RINL internship project demonstration
- B.Tech CSE academic project
- Industrial delay analysis prototype
- Predictive maintenance dashboard prototype
- Shop-wise and equipment-wise delay monitoring

---

## Project Guide Context

This project is suitable for demonstrating how delay records can be converted into useful analytics and maintenance insights for steel plant operations.

The application focuses on practical usability, simple execution, and clean dashboard presentation.
