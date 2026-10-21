# Windows equivalent of the Makefile:  .\make.ps1 data   |  .\make.ps1 test
param([Parameter(Position = 0)][string]$Target = "help")
$py = if ($env:PY) { $env:PY } else { "python" }
switch ($Target) {
    "setup"        { & $py -m pip install -r requirements.txt; & $py -m pip install -e . }
    "data"         { & $py -m saafhawa data }
    "data-offline" { & $py -m saafhawa data --offline }
    "train"        { & $py -m saafhawa train }
    "run"          { & $py -m saafhawa run }
    "serve"        { & $py -m saafhawa.pipeline serve }
    "validate"     { & $py -m saafhawa validate }
    "dashboard"    { & $py -m saafhawa.dashboard }
    "space"        { & $py scripts/make_space.py }
    "export-runs"  { & $py scripts/export_prefect_runs.py }
    "test"         { & $py -m pytest -q }
    default        { "targets: setup data data-offline train run serve validate dashboard space export-runs test" }
}
exit $LASTEXITCODE
