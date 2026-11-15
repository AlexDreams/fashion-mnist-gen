# Pipeline complet sur la machine GPU (RTX 3060) : entraînement à budget égal,
# évaluation de développement, balayage de la guidance, figures, test humain.
#
# Usage, depuis la racine du dépôt, environnement .venv déjà installé :
#   powershell -ExecutionPolicy Bypass -File scripts\pipeline_gpu.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\pipeline_gpu.ps1 -SkipTraining
#
# Tout est fait sur la MÊME machine avec le MÊME classifieur, sinon les
# chiffres ne sont pas comparables. Durée estimée sur RTX 3060 : 1 h 30 à 2 h.
# L'évaluation finale (python eval.py) n'est PAS lancée ici : on la lance une
# seule fois, après avoir figé les modèles et les réglages (voir la fin).

param(
    [int]$Epochs = 30,            # même nombre d'époques pour VAE, GAN et diffusion
    [int]$LatentEpochs = 100,     # le débruiteur latent est minuscule : on rapporte son temps
    [switch]$SkipTraining
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$python = ".\.venv\Scripts\python.exe"
New-Item -ItemType Directory -Force runs\logs | Out-Null

function Step([string]$name, [string[]]$arguments) {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    $log = "runs\logs\$name.log"
    & $python @arguments 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { throw "Échec de l'étape '$name' (voir $log)" }
}

Step "tests" @("-m", "pytest", "-q")

if (-not $SkipTraining) {
    Step "classifieur" @("train_classifier.py", "--epochs", "5")
    Step "vae" @("train_vae.py", "--epochs", "$Epochs", "--seed", "42")
    Step "gan" @("train_gan.py", "--epochs", "$Epochs", "--seed", "42")
    Step "diffusion" @("train_diffusion.py", "--epochs", "$Epochs", "--seed", "42",
                       "--checkpoint", "checkpoints\diffusion.pt")
    Step "diffusion_latente" @("train_latent_diffusion.py", "--epochs", "$LatentEpochs",
                               "--seed", "42")
}

# Tableau de développement : 4 niveaux + DDPM 1000 / DDIM 50 / DDIM 10.
Step "eval_dev" @("eval.py", "--dev")

# Effet de la guidance (développement, pas de ligne réelle à recalculer).
Step "guidance_sweep" @("evaluate_protocol.py", "--models", "ddpm", "ldm",
    "--diffusion-configs", "ddim:50:0", "ddim:50:1", "ddim:50:1.5", "ddim:50:2",
    "ddim:50:3", "ddim:50:5",
    "--evaluation-partition", "development", "--seeds", "2026", "2027", "2028",
    "--no-real-reference", "--output-dir", "runs\guidance_sweep")

Step "figures" @("make_figures.py", "--with-ddpm")

if (-not (Test-Path runs\human_test\manifest.json)) {
    Step "test_humain" @("human_test.py", "prepare")
}

# Archive à partager (checkpoints + résultats + pool du test humain).
$archive = "fashion_mnist_checkpoints.zip"
Compress-Archive -Force -Path checkpoints, runs\eval_dev, runs\guidance_sweep, `
    runs\figures, runs\human_test, runs\gan, runs\vae, runs\diffusion, `
    runs\latent_diffusion -DestinationPath $archive
Write-Host "`nTerminé. Archive à partager : $archive" -ForegroundColor Green
Write-Host @"
Étapes suivantes :
  1. Interface et test humain : .\.venv\Scripts\streamlit.exe run app.py
     (5 personnes, onglet « Test humain »), puis : python human_test.py score
  2. Une fois les modèles et réglages figés : python eval.py   (une seule fois)
"@
