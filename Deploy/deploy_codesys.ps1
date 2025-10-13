# 1. Online > Create Boot Application
# 2. Save Boot application in V:\Userdoc\Mechatronics\Applications\CODESYS\Programs\Main\F.A.T Rig\Application
# 3. To use run .\Deploy\deploy_codesys.ps1 or V:\Userdoc\Mechatronics\Applications\CODESYS\Programs\Main\F.A.T Rig\Application\Deploy\deploy_codesys.ps1 if not in the current folder

param(
    [string]$SourceDir       = "V:\Userdoc\Mechatronics\Applications\CODESYS\Programs\Main\F.A.T Rig\Application",
    [string]$RemotePath      = "/var/opt/codesys/PlcLogic/DLS/",
    [string]$VisuSourceDir   = "V:\Userdoc\Mechatronics\Applications\CODESYS\Programs\Main\F.A.T Rig\Application\PlcLogic\visu",
    [string]$VisuRemotePath  = "/var/opt/codesys/PlcLogic/visu",
    [string]$HostsFile       = ".\Deploy\hosts.txt",
    [string]$User            = "root",
    [string]$Key             = "$env:USERPROFILE\.ssh\id_ed25519",  # adjust if you use a different key
    [string]$ServiceName     = "codesyscontrol.service"             # change if your runtime uses a different name
)

$ErrorActionPreference = "Stop"

# Validate local paths
if (-not (Test-Path -LiteralPath $SourceDir))    { throw "SourceDir not found: $SourceDir" }
if (-not (Test-Path -LiteralPath $VisuSourceDir)){ Write-Warning "VisuSourceDir not found: $VisuSourceDir (skipping visu copy)"; $CopyVisu = $false } else { $CopyVisu = $true }

# Ensure trailing backslash so we copy contents (for the .app/.crc selection)
if (-not $SourceDir.EndsWith("\")) { $SourceDir += "\" }

# Load hosts from file (ignore empty or commented lines)
$Targets = Get-Content -LiteralPath $HostsFile | Where-Object { $_ -and $_ -notmatch '^\s*#' }

foreach ($Target in $Targets) {
    Write-Host "==> Deploying to $Target"

    $DlsDest  = "$User@$Target`:$RemotePath"
    $VisuDest = "$User@$Target`:$VisuRemotePath/"

    # Copy .app and .crc files
    $Files = Get-ChildItem -Path (Join-Path $SourceDir '*') -Include *.app, *.crc -File -ErrorAction Stop
    foreach ($File in $Files) {
        Write-Host "   -> SCP $($File.Name) to $DlsDest"
        scp -i $Key -C -q "$($File.FullName)" "$DlsDest"
    }

    # Copy visu folder contents (if present)
    if ($CopyVisu) {
        Write-Host "   -> Ensuring remote visu path: $VisuRemotePath"
        ssh -i $Key "$User@$Target" "mkdir -p '$VisuRemotePath'"

        # Copy the CONTENTS of the local visu directory to the remote visu directory.
        # Using '\.' ensures contents are copied even with spaces in the path.
        Write-Host "   -> SCP visu contents to $VisuDest"
        scp -i $Key -C -r -q "$VisuSourceDir\." "$VisuDest"
    }

    # Restart Codesys service
    Write-Host "   -> Restarting $ServiceName"
    ssh -i $Key "$User@$Target" "sudo systemctl restart $ServiceName"
}

Write-Host "All deployments complete."