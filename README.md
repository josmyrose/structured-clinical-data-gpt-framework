

# 🧬 Digital Clinical ML: AI-based Digital Twin Models for Structured Clinical Data
**Author:** Josmy Mathew  
**Date:** November 2025  

---

## 🌟 Overview
This project develops an **AI-based digital twin framework** for structured clinical data.  
Using transformer (GPT-style) architectures, the goal is to model patient-level time-series data for:

- 🔮 Predictive analytics (forecasting future lab/vital values)  
- 🧩 Imputation of missing clinical data  
- 🧠 Generation of realistic synthetic patient records  

The work focuses on **lung-related disease cohorts** and demonstrates how large language model (LLM) principles can be adapted to **numeric biomedical data**.

---

## 📊 Dataset Structure
| Table | Description | Key Columns |
|:-------|:-------------|:-------------|
| **Patients** | Demographics and static variables | Id, BIRTHDATE, GENDER, RACE, INCOME |
| **Encounters** | Hospital visits | Id, PATIENT, START, STOP, DESCRIPTION |
| **Observations** | Labs / vitals | DATE, PATIENT, ENCOUNTER, VALUE, UNITS |
| **Conditions** | Diagnoses | START, STOP, PATIENT, DESCRIPTION |

---

## ⚙️ Project Workflow
1. **Data preparation & datetime parsing**  
2. **Merge encounters, observations, and conditions**  
3. **Construct patient-level event sequences**  
4. **Encode features, numeric values, and time gaps**  
5. **Train transformer (MiniGPT) for imputation & forecasting**  
6. **Evaluate and generate synthetic trajectories**

<p align="center">
  <img src="assets/workflow_diagram.png" width="650">
</p>

---

## 📚 Repository Structure
