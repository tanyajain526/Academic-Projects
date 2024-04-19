This project stemmed from the exploration of a hospital patients dataset with the aim of extracting meaningful insights and providing actionable recommendations for hospital management. Leveraging various data analysis techniques, we uncovered valuable patterns and trends within the data, shedding light on factors influencing patient outcomes. Furthermore, by implementing machine learning algorithms, we were able to predict patient survival rates with a commendable accuracy of 80%.
Hospital-Patients Data Set
 
The ‘hospital.csv’ data set, (91713 rows and 85 columns)
 
Each row in this dataset represents a single patient's information during their stay in a healthcare unit. Each row provides a collection of data points (values) for that specific patient, and these data points are organized into columns as described in the column descriptions provided earlier.
For example, the first row contains information about a specific patient, including their unique encounter ID, patient ID, hospital ID, age, BMI, elective surgery status, ethnicity, gender, height, ICU admission source, ICU ID, ICU stay type, ICU type, pre-ICU length of stay, weight, diagnosis codes, clinical measurements, comorbidity indicators, and other relevant information.
Each subsequent row in the dataset represents data for another patient's stay in the healthcare unit, with values for each of the 85 columns providing details about that specific patient's characteristics, clinical parameters, and diagnosis which are explained below:
 
1. encounter_id: Unique identifier associated with a patient unit stay.
2. patient_id: Unique identifier associated with a patient.
3. hospital_id: Unique identifier associated with a hospital.
4. age: The age of the patient on unit admission.
5. bmi: The body mass index of the person on unit admission.
6. elective_surgery: Whether the patient was admitted to the hospital for an elective surgical operation.
7.  ethnicity: The common national or cultural tradition to which the person belongs.
8. gender: Sex of the patient.
9. height: The height of the person on unit admission.
10. icu_admit_source: The location of the patient prior to being admitted to the unit.
11. icu_id: A unique identifier for the unit to which the patient was admitted.
12. icu_stay_type: A classification indicating the type of care the unit can provide.
13. icu_type: A classification indicating the type of care the unit is capable of providing.
14. pre_icu_los_days: The length of stay of the patient between hospital admission and unit admission.
15.  weight: The weight (body mass) of the person on unit admission.
16. apache_2_diagnosis: The APACHE II diagnosis for the ICU admission.
17. apache_3j_diagnosis: The APACHE III-J sub-diagnosis code that best describes the reason for the ICU admission.
18.  apache_post_operative: The APACHE operative status; 1 for post-operative, 0 for non-operative.
19. arf_apache: Whether the patient had acute renal failure during the first 24 hours of their unit stay.
20.gcs_eyes_apache: The eye-opening component of the Glasgow Coma Scale measured during the first 24 hours.
21. gcs_motor_apache: The motor component of the Glasgow Coma Scale measured during the first 24 hours.
22.gcs_unable_apache: Whether the Glasgow Coma Scale was unable to be assessed due to patient sedation.
23. gcs_verbal_apache: The verbal component of the Glasgow Coma Scale measured during the first 24 hours.
24.heart_rate_apache: The heart rate measured during the first 24 hours.
25.intubated_apache: Whether the patient was intubated at the time of the highest scoring arterial blood gas used in the oxygenation score.
26.map_apache: The mean arterial pressure measured during the first 24 hours.
27.resprate_apache: The respiratory rate measured during the first 24 hours.
28. temp_apache: The temperature measured during the first 24 hours.
29. ventilated_apache: Whether the patient was invasively ventilated at the time of the highest scoring arterial blood gas using the oxygenation scoring algorithm.
30. d1_diasbp_max: The patient's highest diastolic blood pressure during the first 24 hours of their unit stay.
31.  d1_diasbp_min: The patient's lowest diastolic blood pressure during the first 24 hours of their unit stay.
32. d1_diasbp_noninvasive_max: The patient's highest diastolic blood pressure during the first 24 hours, non-invasively measured.
33. d1_diasbp_noninvasive_min: The patient's lowest diastolic blood pressure during the first 24 hours, non-invasively measured.
34. d1_heartrate_max: The patient's highest heart rate during the first 24 hours.
35.d1_heartrate_min: The patient's lowest heart rate during the first 24 hours.
36.d1_mbp_max: The patient's highest mean blood pressure during the first 24 hours.
37.d1_mbp_min: The patient's lowest mean blood pressure during the first 24 hours.
38. d1_mbp_noninvasive_max: The patient's highest mean blood pressure during the first 24 hours, non-invasively measured.
39.d1_mbp_noninvasive_min: The patient's lowest mean blood pressure during the first 24 hours, non-invasively measured.
40. d1_resprate_max: The patient's highest respiratory rate during the first 24 hours.
41. d1_resprate_min: The patient's lowest respiratory rate during the first 24 hours.
42. d1_spo2_max: The patient's highest peripheral oxygen saturation during the first 24 hours.
43.  d1_spo2_min: The patient's lowest peripheral oxygen saturation during the first 24 hours.
44.d1_sysbp_max: The patient's highest systolic blood pressure during the first 24 hours.
45.d1_sysbp_min: The patient's lowest systolic blood pressure during the first 24 hours.
46.d1_sysbp_noninvasive_max: The patient's highest systolic blood pressure during the first 24 hours, non-invasively measured.
47.d1_sysbp_noninvasive_min: The patient's lowest systolic blood pressure during the first 24 hours, non-invasively measured.
48.d1_temp_max: The patient's highest core temperature during the first 24 hours, invasively measured.
49.d1_temp_min: The patient's lowest core temperature during the first 24 hours.
50.h1_diasbp_max: The patient's highest diastolic blood pressure during the first hour of their unit stay.
51. h1_diasbp_min: The patient's lowest diastolic blood pressure during the first hour of their unit stay.
52.h1_diasbp_noninvasive_max: The patient's highest diastolic blood pressure during the first hour, invasively measured.
53.h1_diasbp_noninvasive_min: The patient's lowest diastolic blood pressure during the first hour, invasively measured.
54.h1_heartrate_max: The patient's highest heart rate during the first hour of their unit stay.
55.h1_heartrate_min: The patient's lowest heart rate during the first hour of their unit stay.
56.h1_mbp_max: The patient's highest mean blood pressure during the first hour.
57.h1_mbp_min: The patient's lowest mean blood pressure during the first hour.
58.h1_mbp_noninvasive_max: The patient's highest mean blood pressure during the first hour, non-invasively measured.
59.h1_mbp_noninvasive_min: The patient's lowest mean blood pressure during the first hour, non-invasively measured.
60.h1_resprate_max: The patient's highest respiratory rate during the first hour.
61. h1_resprate_min: The patient's lowest respiratory rate during the first hour.
62.h1_spo2_max: The patient's highest peripheral oxygen saturation during the first hour.
63.h1_spo2_min: The patient's lowest peripheral oxygen saturation during the first hour.
64.h1_sysbp_max: The patient's highest systolic blood pressure during the first hour, either non-invasively or invasively measured.
65.h1_sysbp_min: The patient's lowest systolic blood pressure during the first hour, either non-invasively or invasively measured.
66.h1_sysbp_noninvasive_max: The patient's highest systolic blood pressure during the first hour, non-invasively measured.
67.h1_sysbp_noninvasive_min: The patient's lowest systolic blood pressure during the first hour, non-invasively measured.
68.d1_glucose_max: The highest glucose concentration of the patient in their serum or plasma during the first 24 hours.
69.d1_glucose_min: The lowest glucose concentration of the patient in their serum or plasma during the first 24 hours.
70.d1_potassium_max: The highest potassium concentration for the patient in their serum or plasma during the first 24 hours.
71. d1_potassium_min: The lowest potassium concentration for the patient in their serum or plasma during the first 24 hours.
72.apache_4a_hospital_death_prob: The APACHE IVa probabilistic prediction of in-hospital mortality for the patient.
73.apache_4a_icu_death_prob: The APACHE IVa probabilistic prediction of ICU mortality for the patient.
74.aids: Whether the patient has a definitive diagnosis of acquired immune deficiency syndrome (AIDS).
75.cirrhosis: Whether the patient has a history of heavy alcohol use with portal hypertension and varices, other causes of cirrhosis with evidence of portal hypertension and varices, or biopsy proven cirrhosis.
76.diabetes_mellitus: Whether the patient has been diagnosed with diabetes, either juvenile or adult onset, which requires medication.
77.hepatic_failure: Whether the patient has cirrhosis and additional complications, including jaundice and ascites, upper GI bleeding, hepatic encephalopathy, or coma.
78.immunosuppression: Whether the patient has their immune system suppressed within six months prior to ICU admission for any of the following reasons: radiation therapy, chemotherapy, use of non-cytotoxic immunosuppressive drugs, high dose steroids (at least 0.3 mg/kg/day of methylprednisolone or equivalent for at least 6 months).
79.leukemia: Whether the patient has been diagnosed with acute or chronic myelogenous leukemia, acute or chronic lymphocytic leukemia, or multiple myeloma.
80.lymphoma: Whether the patient has been diagnosed with non-Hodgkin lymphoma.
81.  solid_tumor_with_metastasis: Whether the patient has been diagnosed with any solid tumor carcinoma (including malignant melanoma) which has evidence of metastasis.
82.apache_3j_bodysystem: Admission diagnosis group for APACHE III.
83. apache_2_bodysystem: Admission diagnosis group for APACHE II.
84.Column need to be dropped (null column)
85.hospital_death: Whether the patient died during this hospitalization.
 
 
