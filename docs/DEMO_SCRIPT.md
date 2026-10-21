# Demo video script (target 2 min 40 s, limit 3 min)

Record the phone screen for the app and the laptop for the rest. Say the numbers exactly as written; they come from `reports/metrics.json`.

| Time | Show | Say |
|---|---|---|
| 0:00 to 0:15 | Photo or title card: a rider in Lahore smog (or just the PeshGo title) | "Smog advisories assume you can stay indoors. Delivery riders, drivers, construction workers and children can't. PeshGo tells them what tomorrow's air means for them." |
| 0:15 to 0:55 | Phone, the live app. Switch Workers, then Parents and schools. Tap the Urdu button. Scroll to the hour-by-hour strip. Tap Share on WhatsApp | "One screen. A forecast for the next 24 hours, a likely range, and what to do, in English or Urdu. The advice follows the US EPA guide, which lists outdoor workers among the most exposed. The strip shows the cleaner hours of the last two weeks, and one tap shares it." |
| 0:55 to 1:25 | Laptop: Prefect UI with the hourly runs; `manifest.csv`; the Approvals tab | "Behind it, a pipeline runs every hour. Every raw file is stored unchanged with a SHA-256 hash. A data-quality agent fixes, fills or flags problems and logs why, and risky decisions, like dropping a sensor, wait here for a person to approve." |
| 1:25 to 2:00 | The Evidence section: model against baselines | "On data the model never saw, the last-24-hours baseline I declared up front has an error of 18.28. The satellite forecast, CAMS, has 38.98. PeshGo has 19.19: -3.7% better than the baseline and 50.8% better than CAMS. The likely range holds 92% of outcomes. The app also shows the errors by group, for example Hazardous hours (n = 32978): MAE 51.2, worse than the baseline (37.1)." |
| 2:00 to 2:25 | Local app, Ask tab: "Is smog good for my health?", then "What is the capital of France?", then an Urdu question | "Ask it something false and it corrects you, with the source. Ask it something outside its documents and it says I don't know. Forecast numbers come only from stored data rows, never from a language model." |
| 2:25 to 2:40 | Title card with the repo and live link | "Limits: the forecast is a city-wide average, the Urdu needs a native reader, and training history is short. Next: more sensors and a field test with real workers. PeshGo, پیشگو." |

## Before recording
- Open the live app on the phone: https://firstladyemaan-peshgo.static.hf.space/
- Run `make dashboard` for the Ask tab and Approvals tab, `make serve` for the Prefect UI at http://127.0.0.1:4200.
