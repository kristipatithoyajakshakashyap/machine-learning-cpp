param([string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot), [switch]$RequireFullML)
$ErrorActionPreference = 'Stop'
$projectRoot = $ProjectRoot
$files = @(Get-ChildItem -LiteralPath "$projectRoot/01_get_to_know", "$projectRoot/02_data_science", "$projectRoot/03_ml_course" -Recurse -File | Where-Object { $_.FullName -match '[\\/]results[\\/]' })
if ($files.Count -eq 0) { throw 'No local lesson results found. Run lessons first.' }
foreach ($file in $files) {
    if ($file.Length -eq 0) { throw "Empty artifact: $($file.FullName)" }
    if ($file.Extension -eq '.json') { Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json | Out-Null }
}
Write-Output "Validated $($files.Count) nonempty lesson-local artifacts; JSON parses successfully."
if ($RequireFullML) {
    $workflows = @(Get-ChildItem -LiteralPath "$projectRoot/03_ml_course" -Recurse -File -Filter '*end_to_end.cpp')
    foreach ($workflow in $workflows) {
        $resultsRoot = Join-Path $workflow.DirectoryName 'results'
        foreach ($folder in (Get-ChildItem -LiteralPath $resultsRoot -Directory)) {
            if ($folder.Name -notmatch '^[A-Za-z0-9_]+_results$') { throw "Result folder must identify its C++ source: $($folder.FullName)" }
        }
        foreach ($name in @('README.md', 'theory.md', 'math_intuition.md', 'implementation.md', 'exercises.md', 'CMakeLists.txt')) {
            if (-not (Test-Path -LiteralPath (Join-Path $workflow.DirectoryName $name) -PathType Leaf)) { throw "Missing teaching file $name in $($workflow.DirectoryName)" }
        }
        $run = Join-Path $workflow.DirectoryName ('results/' + $workflow.BaseName + '_results')
        if (Test-Path -LiteralPath (Join-Path $run 'full')) { $run = Join-Path $run 'full' }
        foreach ($relative in @('run_manifest.json', 'report.md', 'execution.log', 'data/schema.json', 'eda/findings.md', 'evaluation/metrics.json', 'inference/reload_verification.json')) {
            if (-not (Test-Path -LiteralPath (Join-Path $run $relative) -PathType Leaf)) { throw "Missing full-run artifact: $run/$relative" }
        }
        $selection = Test-Path -LiteralPath (Join-Path $run 'validation/candidate_scores.csv') -PathType Leaf
        $calibration = (Test-Path -LiteralPath (Join-Path $run 'validation/calibration_scores.csv') -PathType Leaf) -and (Test-Path -LiteralPath (Join-Path $run 'validation/threshold.json') -PathType Leaf)
        if (-not $selection -and -not $calibration) { throw "Missing model-selection or threshold-calibration record: $run" }
        $models = @(Get-ChildItem -LiteralPath (Join-Path $run 'model') -File -Filter '*.txt')
        if ($models.Count -eq 0) { throw "Missing saved C++ model state: $run" }
        $reload = Get-Content -LiteralPath (Join-Path $run 'inference/reload_verification.json') -Raw | ConvertFrom-Json
        $errorProperty = $reload.PSObject.Properties['max_absolute_error']
        $numericMatch = $null -ne $errorProperty -and $null -ne $errorProperty.Value -and [double]$errorProperty.Value -ge 0 -and [double]$errorProperty.Value -le 1e-9
        if ($reload.passed -ne $true -and -not $numericMatch) { throw "Model reload not verified: $run" }
        $manifest = Get-Content -LiteralPath (Join-Path $run 'run_manifest.json') -Raw | ConvertFrom-Json
        if ($manifest.quick -eq $true) { throw "Quick run cannot certify full artifacts: $run" }
    }
    Write-Output "Verified teaching files and required full-run artifacts for all $($workflows.Count) ML modules."
}
