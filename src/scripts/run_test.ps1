# Test Recognition Script
# 测试人脸识别准确率

param(
    [string]$Mode = "both"  # frontal, profile, or both
)

Write-Host "Starting recognition testing..." -ForegroundColor Green
Write-Host ""

# Set paths
$MODEL_PATH = "..\..\models\best_model.pth"
$GALLERY_PATH = "..\..\galleries\gallery_features.pth"
$OCCLUDED_ROOT = "..\..\occluded"

Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  Model: $MODEL_PATH" -ForegroundColor White
Write-Host "  Gallery: $GALLERY_PATH" -ForegroundColor White
Write-Host "  Occluded Dataset: $OCCLUDED_ROOT" -ForegroundColor White
Write-Host "  Test Mode: $Mode" -ForegroundColor White
Write-Host ""

# Check if files exist
if (-not (Test-Path $MODEL_PATH)) {
    Write-Host "Error: Model file not found at $MODEL_PATH" -ForegroundColor Red
    Write-Host "Please run run_train.ps1 first to train the model." -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path $GALLERY_PATH)) {
    Write-Host "Error: Gallery file not found at $GALLERY_PATH" -ForegroundColor Red
    Write-Host "Please run run_train.ps1 first to build the gallery." -ForegroundColor Yellow
    exit 1
}

# Run testing
python .\test_recognition.py `
    --model-path $MODEL_PATH `
    --gallery-path $GALLERY_PATH `
    --occluded-root $OCCLUDED_ROOT `
    --mode $Mode

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✓ Testing completed successfully!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "✗ Testing failed!" -ForegroundColor Red
}
