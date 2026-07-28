$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$templateDir = Join-Path $root "Template_for_ICLR_2025_Conference_Submission"
$tectonic = Join-Path $root "_tectonic\tectonic.exe"
$paperStem = "remap_former_current_paper_iclr2025"
$outputDir = Join-Path $root "output\pdf"

Push-Location $root
try {
    python "make_remap_former_current_paper_figures.py"
} finally {
    Pop-Location
}

Push-Location $templateDir
try {
    & $tectonic "$paperStem.tex" --keep-logs --keep-intermediates
    if ($LASTEXITCODE -ne 0) {
        throw "Tectonic failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $templateDir "$paperStem.pdf") -Destination (Join-Path $outputDir "$paperStem.pdf") -Force

Write-Host "Built: $(Join-Path $outputDir "$paperStem.pdf")"
