# PeshGo. On Windows without make, use: .\make.ps1 <target>
PY ?= python

.PHONY: setup data data-offline train run serve validate dashboard space test export-runs clean

setup:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

data:            ## fetch everything, verify the manifest, build cleaned + gold layers
	$(PY) -m saafhawa data

data-offline:    ## rebuild from the raw files already listed in the manifest (no network)
	$(PY) -m saafhawa data --offline

train:
	$(PY) -m saafhawa train

run:             ## one supervised pipeline run
	$(PY) -m saafhawa run

serve:           ## the hourly Prefect schedule (5 minutes past every hour)
	$(PY) -m saafhawa.pipeline serve

validate:        ## compare with Punjab EPD daily AQI reports
	$(PY) -m saafhawa validate

dashboard:
	$(PY) -m saafhawa.dashboard

space:
	$(PY) scripts/make_space.py

export-runs:
	$(PY) scripts/export_prefect_runs.py

test:
	$(PY) -m pytest -q

clean:
	rm -rf data models build reports/*.parquet
