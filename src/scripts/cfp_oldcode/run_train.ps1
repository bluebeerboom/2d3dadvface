# Train and Build Gallery Script
# 训练模型并建立特征库

Write-Host "Starting training and gallery building..." -ForegroundColor Green
Write-Host ""

# Set paths
$CFP_ROOT = "..\..\cfp-dataset"
$EPOCHS = 30
$BATCH_SIZE = 32
$LR = 0.01
$SAVE_DIR = "..\..\models"
$GALLERY_PATH = "..\..\galleries\gallery_features.pth"

Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  CFP Dataset: $CFP_ROOT" -ForegroundColor White
Write-Host "  Epochs: $EPOCHS" -ForegroundColor White
Write-Host "  Batch Size: $BATCH_SIZE" -ForegroundColor White
Write-Host "  Learning Rate: $LR" -ForegroundColor White
Write-Host "  Save Directory: $SAVE_DIR" -ForegroundColor White
Write-Host "  Gallery Path: $GALLERY_PATH" -ForegroundColor White
Write-Host ""

# Run training
python .\train_and_build_gallery.py `
    --cfp-root $CFP_ROOT `
    --epochs $EPOCHS `
    --batch-size $BATCH_SIZE `
    --lr $LR `
    --save-dir $SAVE_DIR `
    --gallery-path $GALLERY_PATH

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✓ Training completed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next step: Run test_recognition.ps1 to test on occluded dataset" -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "✗ Training failed!" -ForegroundColor Red
}
