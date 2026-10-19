*Source 1: name, publisher, URL, licence, update frequency*
Open-Meteo Historical Weather API, Open-Meteo GmbH, https://open-meteo.com/en/docs/historical-weather-api , CC BY 4.0, hourly, non-commercial free. Real-time source listed under Climate and heat-health track.

*Source 2: name, publisher, URL, licence, update frequency*
OpenAQ Air Quality API, OpenAQ Inc, https://docs.openaq.org/ , CC BY 4.0, hourly. Real-time source listed under Air quality track. Covers Quetta, Karachi stations.

*Which source is not on the track list, and why is it credible?*
Pakistan Bureau of Statistics (PBS), https://www.pbs.gov.pk, Publisher: Government of Pakistan, Licence: Government Open Data. From "Pakistani sources to check first" table in guide. AND Pakistan Meteorological Department (PMD), https://www.pmd.gov.pk, Publisher: PMD, for heat advisory validation. Both are primary publishers, not Kaggle re-uploads, cover Pakistan districts, directly relevant for age 60+ population and heat-health.

*Expected volume*
Open-Meteo: 2 years hourly for 3 cities (Quetta, Karachi, Lahore) = ∼52,560 rows. OpenAQ: 2 years daily PM2.5 = ∼2,190 rows. PBS: 154 districts age structure. Total >55k rows. Meets minimum volume.

*Why this data fits the problem and the primary user*
Problem: Dementia risk in 60+ population triggered by prolonged heat and air pollution causing neuroinflammation - leading to late diagnosis in Pakistan due to <5 neurologists per million. Primary user: Family caregiver / BHU doctor in Quetta deciding on early screening. Heat index + PM2.5 are modifiable risk factors per WHO Dementia Risk Reduction Guideline.

*Baseline you declare (must be fixed before you see results)*
Baseline: Next week's dementia risk zone = mean risk of last 4 weeks for same district. Model must beat this by 15% to be valid.
