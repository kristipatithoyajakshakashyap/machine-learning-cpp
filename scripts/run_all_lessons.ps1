param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$Filter = '',
    [switch]$SkipMlFull
)
$ErrorActionPreference = 'Stop'
$projectRoot = $ProjectRoot
$manifestPath = Join-Path $projectRoot 'build/lesson_manifest.txt'
if (-not (Test-Path -LiteralPath $manifestPath)) { throw "Missing $manifestPath. Configure and build first." }
$compiler = Get-Command g++ -ErrorAction Stop
$env:PATH = (Split-Path -Parent $compiler.Source) + ';' + $env:PATH

$textInference = Join-Path $projectRoot '03_ml_course/05_text_classification/data/inference_messages.txt'
$penguinInference = Join-Path $projectRoot '03_ml_course/02_unsupervised/tests/penguin_inference.csv'

function Invoke-Lesson {
    param([string]$Target, [string]$Exe, [string]$ResultsDir, [string[]]$Arguments, [string]$StdIn)
    New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $quoted = ($Arguments | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' '
    $stdinFile = $null
    if ($StdIn) {
        $stdinFile = [System.IO.Path]::GetTempFileName()
        [System.IO.File]::WriteAllBytes($stdinFile, [System.Text.Encoding]::ASCII.GetBytes($StdIn))
        $psi.FileName = 'cmd.exe'
        $psi.Arguments = '/c ""' + $Exe + '" ' + $quoted + ' < "' + $stdinFile + '""'
    } else {
        $psi.FileName = $Exe
        $psi.Arguments = $quoted
    }
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.WorkingDirectory = $ResultsDir
    $proc = [System.Diagnostics.Process]::Start($psi)
    $stdout = $proc.StandardOutput.ReadToEndAsync()
    $stderr = $proc.StandardError.ReadToEndAsync()
    $proc.WaitForExit()
    if ($stdinFile) { Remove-Item -LiteralPath $stdinFile -Force -ErrorAction SilentlyContinue }
    if ($proc.ExitCode -ne 0) {
        if ($stdout.Result) { Write-Output $stdout.Result }
        if ($stderr.Result) { Write-Output $stderr.Result }
    }
    if (-not (Get-ChildItem -LiteralPath $ResultsDir -Force | Select-Object -First 1)) {
        Remove-Item -LiteralPath $ResultsDir -Force
        $parent = Split-Path -Parent $ResultsDir
        if (-not (Get-ChildItem -LiteralPath $parent -Force | Select-Object -First 1)) { Remove-Item -LiteralPath $parent -Force }
    }
    return $proc.ExitCode
}

$lines = Get-Content -LiteralPath $manifestPath | Where-Object { $_.Trim() -ne '' }
$entries = foreach ($line in $lines) {
    $parts = $line.Split('|')
    if ($parts.Count -ne 3) { throw "Malformed manifest line: $line" }
    [pscustomobject]@{ Target = $parts[0]; Exe = $parts[1]; Results = $parts[2] }
}
if ($Filter) { $entries = $entries | Where-Object { $_.Target -match $Filter } }
$ordered = @($entries | Where-Object { $_.Target -notmatch 'predict$' }) + @($entries | Where-Object { $_.Target -match 'predict$' })

$failures = @()
$count = 0
foreach ($e in $ordered) {
    $count++
    if (-not (Test-Path -LiteralPath $e.Exe)) { $failures += "$($e.Target): executable not built ($($e.Exe))"; continue }
    $runs = @()
    $isMl = $e.Results -match '[\\/]03_ml_course[\\/]'
    if ($e.Target -eq 'conc_user_input') {
        $runs += @{ Args = @(); StdIn = "Alice`n42`nthe matrix is cool`n25`n" }
    } elseif ($e.Target -eq 's11_predict') {
        $capstone = Join-Path (Split-Path -Parent $e.Results) '05_eval_model_results'
        $runs += @{ Args = @((Join-Path $capstone 'model_state.txt'), (Join-Path $capstone 'test_features.csv'), (Join-Path $capstone 'predictions.csv')); StdIn = '' }
    } elseif ($isMl -and $e.Target -match 'predict$') {
        $moduleResults = Split-Path -Parent $e.Results
        $e2e = @(Get-ChildItem -LiteralPath $moduleResults -Directory | Where-Object { $_.Name -match '^\d+_end_to_end_results$' })
        if ($e2e.Count -ne 1) { $failures += "$($e.Target): expected one end_to_end results dir in $moduleResults"; continue }
        $full = Join-Path $e2e[0].FullName 'full'
        if ($e.Results -match '[\\/]04_time_series[\\/]') {
            $modelFile = switch -Regex ($e.Results) {
                '01_forecasting_baselines'     { 'seasonal_model.txt'; break }
                '02_lag_feature_regression'    { 'lag_model.txt'; break }
                '03_rolling_origin_evaluation' { 'lag_model.txt'; break }
                '04_exponential_smoothing'     { 'holt_winters_model.txt'; break }
                '05_arima'                     { 'arima_model.txt'; break }
                default                        { 'lag_model.txt' }
            }
            $runs += @{ Args = @('--model', (Join-Path $full "model/$modelFile")); StdIn = '' }
        } elseif ($e.Results -match '[\\/]05_text_classification[\\/]') {
            $runs += @{ Args = @('--predict', $textInference, '--model', (Join-Path $full 'model')); StdIn = '' }
        } elseif ($e.Results -match '[\\/]06_recommender_systems[\\/]') {
            $runs += @{ Args = @('--model', (Join-Path $full 'model/model.txt'), '--user', '1'); StdIn = '' }
        } elseif (Test-Path -LiteralPath $full) {
            $runs += @{ Args = @('--predict', (Join-Path $full 'data/holdout_features.csv'), '--model', (Join-Path $full 'model')); StdIn = '' }
        } else {
            $runs += @{ Args = @($penguinInference); StdIn = '' }
        }
    } elseif ($isMl -and $e.Target -match 'end_to_end$') {
        if ($SkipMlFull) { $runs += @{ Args = @('--quick'); StdIn = '' } }
        else { $runs += @{ Args = @(); StdIn = '' }; $runs += @{ Args = @('--quick'); StdIn = '' } }
    } else {
        $runs += @{ Args = @(); StdIn = '' }
    }
    foreach ($r in $runs) {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $code = Invoke-Lesson -Target $e.Target -Exe $e.Exe -ResultsDir $e.Results -Arguments $r.Args -StdIn $r.StdIn
        $sw.Stop()
        $argText = if ($r.Args.Count) { ' ' + ($r.Args -join ' ') } else { '' }
        Write-Output ("[{0,3}/{1}] {2}{3} -> exit {4} ({5:n1}s)" -f $count, $ordered.Count, $e.Target, $argText, $code, $sw.Elapsed.TotalSeconds)
        if ($code -ne 0) {
            $failures += "$($e.Target)$argText exited with $code"
            $e2eQuick = ($e.Target -match 'end_to_end$') -and ($r.Args -contains '--quick')
            if (-not $e2eQuick) { break }
        }
    }
}
Write-Output ''
if ($failures.Count) {
    Write-Output "FAILED ($($failures.Count)):"
    $failures | ForEach-Object { Write-Output "  $_" }
    exit 1
}
Write-Output "All $($ordered.Count) lessons ran successfully."
