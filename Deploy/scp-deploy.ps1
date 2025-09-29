# To use run .\Deploy\scp-deploy.ps1 or V:\Userdoc\Mechatronics\Applications\Python\R&D AutoChart Generation\Deploy\scp-deploy.ps1 if not in the current folder

param(
    [string]$SourceDir = "V:\Userdoc\Mechatronics\Applications\Python\R&D AutoChart Generation\DLS Chart Generation",
    [string]$RemotePath = "/var/opt/codesys/PlcLogic/python/",
    [string]$HostsFile = ".\Deploy\hosts.txt",
    [string]$User = "root",
    [string]$Key = "$env:USERPROFILE\.ssh\id_ed25519"   # adjust if you use a different key
)

$ErrorActionPreference = "Stop"

# Ensure trailing backslash so we copy contents
if (-not $SourceDir.EndsWith("\")) { $SourceDir += "\" }

# Load hosts from file (ignore empty or commented lines)
$Targets = Get-Content $HostsFile | Where-Object { $_ -and $_ -notmatch '^\s*#' }

foreach ($Target in $Targets) {
    Write-Host "==> Copying to $Target"
    $Destination = "$User@$Target`:$RemotePath"
    scp -i $Key -r -C "$SourceDir*" $Destination
}
Write-Host "Done."