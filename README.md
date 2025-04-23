# GUCD_NCS

# Overlapping Community Detection from Mobility Network

This repository provides a full workflow for detecting **overlapping communities** from urban **mobility networks**, using **New York City** as a demo. The process goes from raw data to spatial mapping and socioeconomic analysis.

![Workflow Diagram](workflow.png)

---

## 🧭 Overview

We aim to detect and analyze community structures embedded in human mobility patterns. Unlike traditional partitioning approaches, our method supports **overlapping communities**, which better reflects the complexity of urban life.

This codebase includes four main components:

1. **Data Preprocessing**  
   Prepares raw mobility network data for analysis, output the A matrix and X matrix required by GCN.

2. **Community Detection**  
   Applies overlapping community detection algorithms to identify mobility-based communities.

3. **Spatial Mapping**  
   Maps community labels to geographic units (CBG level).

4. **Socioeconomic Analysis**  
   Uses external information (e.g., POIs) to interpret the detected communities from a socioeconomic perspective.

---

## 🏙️ New York City as Demo

We select **New York City** as a representative case study.

- We provide a sample dataset (~3,000 rows) for demonstration purposes.
- All final results (community detection, mapping, and analysis) are based on the **full mobility dataset** of NYC.

---

## 📁 Repository Structure

| File | Description |
|------|-------------|
| `NCS_GUCD_data_preprocessing.ipynb` | Output A and X matrix |
| `NCS_GUCD_community_detection.ipynb` | Community detection |
| `NCS_GUCD_mapping_to_spatial.ipynb` | Map community results to CBGs |
| `NCS_GUCD_POI_analysis.ipynb` | Analyze community characteristics using POIs |
| `workflow.png` | Workflow illustration |
| `demo_data/` | Sample input and all necessary outputs to reproduce the results |

---

## 📦 Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
