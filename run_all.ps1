<#
 One-command reproduction on Windows PowerShell (project root, venv active).
   .\run_all.ps1            real run: needs `ollama serve`, llama3.2:3b, nomic-embed-text
   .\run_all.ps1 -Offline   stand-ins only (plumbing, NOT model results; no Ollama)
 Stops at the first failing step. The last real step is the regression gate:
 exit 0 = PASS, 1 = FAIL (gate named in results\regress.md), 2 = invalid comparison.
 Note: regress --compare also needs a session whose safety flicker matches baselines\v1.json (see HANDOFF.md).
#>
param([switch]$Offline)
$ErrorActionPreference = "Stop"
function Step($label, [scriptblock]$cmd) {
  Write-Host "`n=== $label ===" -ForegroundColor Cyan
  & $cmd
  if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $label (exit $LASTEXITCODE)" -ForegroundColor Red; exit $LASTEXITCODE }
}
Step "build data"      { python build_data.py }
Step "unit tests"      { python -m unittest }
if ($Offline) {
  Step "sweep (offline, bm25)" { python cli.py sweep --offline --retrievers bm25 }
  Step "safety (offline)"      { python cli.py safety --offline }
  Step "eval (offline)"        { python cli.py eval --config configs/baseline.yaml --out results/offline_eval.json }
  Write-Host "`nOffline run done: plumbing only." -ForegroundColor Yellow
  exit 0
}
Step "check Ollama + models" { python cli.py check }
Step "golden eval"           { python cli.py eval --out results/baseline.json }
Step "retrieval sweep"       { python cli.py sweep }
Step "safety suite"          { python cli.py safety }
Step "regression gate"       { python cli.py regress --compare baselines/v1.json }
Write-Host "`nDone. Reports are in results\ (git-ignored); frozen tables are in docs\results\." -ForegroundColor Green
