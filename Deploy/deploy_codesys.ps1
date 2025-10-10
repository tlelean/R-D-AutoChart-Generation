# 1. Online > Create Boot Application
# 2. Save Boot application in V:\Userdoc\Mechatronics\Applications\CODESYS\Programs\Main\F.A.T Rig\Application
# 3. To use run .\Deploy\deploy_codesys.ps1 or V:\Userdoc\Mechatronics\Applications\CODESYS\Programs\Main\F.A.T Rig\Application\Deploy\deploy_codesys.ps1 if not in the current folder

param(
    [string]$SourceDir = "V:\Userdoc\Mechatronics\Applications\CODESYS\Programs\Main\F.A.T Rig\Application",
    [string]$RemotePath = "/var/opt/codesys/PlcLogic/DLS/",
    [string]$HostsFile = ".\Deploy\hosts.txt",
    [string]$User = "root",
    [string]$Key = "$env:USERPROFILE\.ssh\id_ed25519",  # adjust if you use a different key
    [string]$ServiceName = "codesyscontrol.service"     # change if your runtime uses a different name
)

$ErrorActionPreference = "Stop"

# Ensure trailing backslash so we copy contents
if (-not $SourceDir.EndsWith("\")) { $SourceDir += "\" }

# Load hosts from file (ignore empty or commented lines)
$Targets = Get-Content $HostsFile | Where-Object { $_ -and $_ -notmatch '^\s*#' }

foreach ($Target in $Targets) {
    Write-Host "==> Deploying to $Target"
    $Destination = "$User@$Target`:$RemotePath"

    $Files = Get-ChildItem -Path "$SourceDir*" -Include *.app, *.crc -File
    foreach ($File in $Files) {
        scp -i $Key -C "$($File.FullName)" "$Destination"
    }

    # Restart Codesys service
    Write-Host "   -> Restarting $ServiceName"
    ssh -i $Key "$User@$Target" "sudo systemctl restart $ServiceName"
}

Write-Host "All deployments complete."