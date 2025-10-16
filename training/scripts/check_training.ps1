# check_training.ps1
# Reproducible YOLO dataset check (compatible with PowerShell 5+)

param(
    [string]$DataYaml = "./configs/datav8_gtsdb.yaml",
    [string]$Model = "yolov8n.pt",
    [int]$ImgSize = 256
)

Write-Host "=== Checking YOLO dataset ==="
Write-Host "Config: $DataYaml | Model: $Model | imgsz: $ImgSize"
Write-Host "----------------------------------------------------"

# --- Read train/val paths manually (no external modules) ---
try {
    $trainPath = (Select-String -Path $DataYaml -Pattern '^\s*train:\s*(.+)$').Matches[0].Groups[1].Value.Trim(" '""")
    $valPath   = (Select-String -Path $DataYaml -Pattern '^\s*val:\s*(.+)$').Matches[0].Groups[1].Value.Trim(" '""")
    Write-Host "Train path: $trainPath"
    Write-Host "Val path:   $valPath"
} catch {
    Write-Error "Could not read train/val paths from YAML file."
    exit 1
}

# --- Run YOLO validation (only checks data loading) ---
try {
    yolo val data=$DataYaml model=$Model imgsz=$ImgSize device=cpu
} catch {
    Write-Warning "YOLO validation failed to start: $($_.Exception.Message)"
}

Write-Host "`n=== Basic structure check ==="

# --- Helper functions for counting ---
function Count-Files($imgPath) {
    if (-not (Test-Path $imgPath)) { return 0 }
    return (Get-ChildItem -Recurse -Include *.jpg,*.png -Path $imgPath -ErrorAction SilentlyContinue).Count
}

function Count-Labels($imgPath) {
    $labelPath = Join-Path (Split-Path $imgPath -Parent) "labels"
    if (-not (Test-Path $labelPath)) { return 0 }
    return (Get-ChildItem -Recurse -Include *.txt -Path $labelPath -ErrorAction SilentlyContinue).Count
}

# --- Count images and labels ---
$trainImages = Count-Files $trainPath
$valImages   = Count-Files $valPath
$trainLabels = Count-Labels $trainPath
$valLabels   = Count-Labels $valPath

Write-Host "Train:  $trainImages images, $trainLabels labels"
Write-Host "Val:    $valImages images, $valLabels labels"
Write-Host "----------------------------------------------------"
Write-Host "If YOLO reports '0 labels found' or these counts are 0,"
Write-Host "the issue is in dataset paths or file structure, not the model."
