# Run All Tests Script
# 运行所有测试模式

Write-Host "Running all recognition tests..." -ForegroundColor Green
Write-Host ""

$ErrorCount = 0

# Test frontal view
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Testing FRONTIAL view..." -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
.\run_test.ps1 -Mode frontal
if ($LASTEXITCODE -ne 0) { $ErrorCount++ }

Write-Host ""
Write-Host ""

# Test profile view
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Testing PROFILE view..." -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
.\run_test.ps1 -Mode profile
if ($LASTEXITCODE -ne 0) { $ErrorCount++ }

Write-Host ""
Write-Host ""

# Test both views
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Testing BOTH views..." -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
.\run_test.ps1 -Mode both
if ($LASTEXITCODE -ne 0) { $ErrorCount++ }

Write-Host ""
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "All tests completed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

if ($ErrorCount -gt 0) {
    Write-Host "$ErrorCount test(s) failed!" -ForegroundColor Red
} else {
    Write-Host "All tests passed successfully!" -ForegroundColor Green
}
