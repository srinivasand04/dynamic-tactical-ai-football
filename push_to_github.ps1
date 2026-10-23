<#
  Push this project to a new GitHub repository: code, settings, figures, tables and presentations go in; data (about 1.7 GB), virtual
  environments and personal files stay out (see .gitignore).

  One-time setup (once per computer):   gh auth login        (choose GitHub.com, HTTPS, log in through the browser)

  Run from the project folder:
      powershell -ExecutionPolicy Bypass -File .\push_to_github.ps1
      powershell -ExecutionPolicy Bypass -File .\push_to_github.ps1 -RepoName my-name -Visibility public

  Safe to run again: it commits only what changed and pushes to the same repository.
#>
param(
    [string]$RepoName = "dynamic-tactical-ai-football",
    [ValidateSet("private", "public")][string]$Visibility = "private",
    [string]$Message = "Dynamic Tactical AI: Key States, graphs, ball-owner graph, pass models, evidence"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Step($t) { Write-Host "`n== $t" -ForegroundColor Cyan }

Step "Checking tools"
git --version
gh --version | Select-Object -First 1
gh auth status 2>$null
if ($LASTEXITCODE -ne 0) { throw "Not logged in to GitHub. Run:  gh auth login   and then run this script again." }

Step "Preparing files"
New-Item -ItemType Directory -Force docs | Out-Null
foreach ($d in @("deck_work\ESILV_Key_States_and_Graphs_8_Slides.pptx", "deck_work\Project_Files_Map_10_Slides.pptx")) {
    if (Test-Path $d) { Copy-Item $d docs\ -Force; Write-Host "copied $d to docs\" }
}
if ((Test-Path requirements-lock.txt) -and -not (Test-Path requirements-grf.txt)) {
    Get-Content requirements-lock.txt | Set-Content -Encoding utf8 requirements-grf.txt      # the original was saved as UTF-16
    Write-Host "wrote requirements-grf.txt (UTF-8 copy of requirements-lock.txt)"
}

Step "Creating the local repository"
if (-not (Test-Path .git)) { git init -b main }
git add -A

Step "Safety check: no file above 50 MB"
$big = git diff --cached --name-only | Where-Object { (Test-Path $_) -and ((Get-Item $_).Length -gt 50MB) }
if ($big) { git reset | Out-Null; throw "These files are too large for GitHub, add them to .gitignore: $($big -join ', ')" }
$n = (git diff --cached --name-only | Measure-Object).Count
Write-Host "$n files will be committed"

Step "Commit"
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) { git commit -m $Message } else { Write-Host "nothing new to commit" }

Step "Create the GitHub repository and push"
gh repo view $RepoName 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "repository exists: pushing to it"
    $url = gh repo view $RepoName --json url --jq .url
    if (-not (git remote | Select-String -Pattern "^origin$")) { git remote add origin $url }
    git push -u origin main
} else {
    gh repo create $RepoName "--$Visibility" --source . --remote origin --push --description "Dynamic Tactical AI for football: Key States, player graphs, ball-owner graph, pass models (DFL tracking data + Google Research Football)"
}
Step "Done"
gh repo view --json url --jq .url
