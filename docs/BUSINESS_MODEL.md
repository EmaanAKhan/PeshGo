# Business model and Pakistan impact

## Who it is for
1. **Schools** (Lahore has thousands of private schools): a daily 6 PM decision — assembly, sports, outdoor breaks.
2. **Parents of children with asthma** and **families with elderly members**.
3. **Clinics and employers** with outdoor workers (traffic police, construction, delivery riders).

## Why they would use it
Existing apps give a single number with no range and no advice. PeshGo gives tomorrow's likely range, the
category in Urdu, and what to do, with the source cited. It works on a phone browser; no install.

## How it is paid for
| Stream | Customer | Price idea |
|---|---|---|
| Free public dashboard + Urdu advice | everyone | free (public good) |
| School plan: 6 PM WhatsApp / SMS alert to parents, PDF notice in Urdu | private school chains | PKR 3,000–5,000 / month per campus |
| API for apps, telcos and delivery platforms | businesses | per-call or monthly |
| Partnership / grant | Punjab EPD, Smog Commission, UNICEF, WHO Pakistan | sponsored public alerts |

## Costs
Runs on free tiers: OpenAQ, Open-Meteo, NASA FIRMS, Hugging Face Spaces, Prefect. One small VM for the hourly flow.
SMS/WhatsApp is the main variable cost.

## Impact (to be measured, not claimed)
- Hours of outdoor exposure avoided by children on "Unhealthy" days at partner schools.
- Forecast skill vs persistence and CAMS, published on the dashboard.
- Agreement with Punjab EPD's official AQI (local validation).

## Risks
Few reference monitors · low-cost sensor humidity bias · advice must defer to official notices · keep the forecast
honest when data is late (the dashboard shows freshness for every source).
