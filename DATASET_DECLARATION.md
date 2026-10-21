*Source 1: name, publisher, URL, licence, update frequency*
OpenAQ API v3 (PM2.5 monitors within 25 km of Lahore), OpenAQ Inc, https://docs.openaq.org/ , licence set by each data provider (recorded in every raw file), hourly. Real-time source listed under the Air quality track.

*Source 2: name, publisher, URL, licence, update frequency*
Open-Meteo (ERA5 weather archive and forecast; Copernicus CAMS air quality), Open-Meteo GmbH, https://open-meteo.com/ , CC BY 4.0, hourly, non-commercial free. Real-time source listed under the Climate and heat-health track.

*Which source is not on the track list, and why is it credible?*
NASA FIRMS, VIIRS active-fire detections (Suomi NPP), NASA LANCE / ESDIS, https://firms.modaps.eosdis.nasa.gov/ , NASA open data, near-real-time. It is a primary publisher (the satellite operator), not a re-upload, with documented algorithms and global coverage; I use fire counts and radiative power upwind of Lahore as a crop-burning signal. Punjab EPD's daily AQI reports (https://epd.punjab.gov.pk/air_quality , Government of the Punjab) are used only to validate my PM2.5 locally, never for training.

*Expected volume*
OpenAQ: 543,283 hourly rows from 78 monitors (2022-09-02 to 2026-10-21). Open-Meteo weather: 36,270 hourly rows; CAMS: 36,270 hourly rows. NASA FIRMS: 761 detections. Well above the 50,000-row minimum from OpenAQ alone.

*Why this data fits the problem and the primary user*
Problem: smog advisories in Lahore assume people can stay indoors, but outdoor workers (riders, drivers, construction workers, vendors), children and schools cannot, and the US EPA guide lists outdoor workers among the groups most exposed. Primary user: an outdoor worker, a parent or a school deciding the evening before whether to shift work or move activities indoors. PM2.5 from ground monitors is the quantity the guidance is written in; weather, satellite-model air quality and upwind fires explain where it is heading.

*Baseline you declare (must be fixed before you see results)*
Baseline: persistence, the mean PM2.5 of the next 24 hours equals the mean of the last 24 hours for the same monitor (`baseline: persistence_24h` in config.yaml). A second reference is the CAMS forecast for the same 24 hours. The time split (train to 31 Oct 2024, validation to 31 Mar 2025, test after, 24 h purge) was fixed in config.yaml before any model was trained, and the test set was scored once.

*Change note*
This repository's first declaration (19 Oct 2026, 11:33 PKT) described a different project, a dementia-risk model for Quetta using Pakistan Bureau of Statistics and PMD data. I changed to the project above afterwards; the earlier text is in the git history. Everything on this page describes the project I actually built.
