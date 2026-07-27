param(
  [string]$WarmupConfig = "config/sef_warmup.yaml",
  [string]$FineTuneConfig = "config/sef_end_to_end.yaml"
)

$ErrorActionPreference = "Stop"
python scripts/preflight.py --config $WarmupConfig
python train.py --config $WarmupConfig
python scripts/preflight.py --config $FineTuneConfig
python train.py --config $FineTuneConfig
