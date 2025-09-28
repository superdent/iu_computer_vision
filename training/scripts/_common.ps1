# training/scripts/_common.ps1
$ErrorActionPreference = "Stop"
$DATA_YAML = (Resolve-Path (Join-Path $PSScriptRoot '..\configs\data.yaml'))
