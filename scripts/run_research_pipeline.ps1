param(
  [string]$WarmupConfig = "config/sef_warmup.yaml",
  [string]$FineTuneConfig = "config/sef_end_to_end.yaml"
)

$ErrorActionPreference = "Stop"
python scripts/preflight.py --config $WarmupConfig
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python train.py --config $WarmupConfig
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python scripts/preflight.py --config $FineTuneConfig
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python train.py --config $FineTuneConfig
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
