param(
    [int]$ApiPort = 8000,
    [int]$FrontendPort = 8501,
    [int]$MlflowPort = 5000,
    [int]$AirflowPort = 8080
)

$ErrorActionPreference = "Stop"

function Get-LanIp {
    $ip = Get-NetIPAddress -AddressFamily IPv4 |
        Where-Object {
            $_.IPAddress -notlike "127.*" -and
            $_.IPAddress -notlike "169.254.*" -and
            $_.PrefixOrigin -ne "WellKnown" -and
            $_.AddressState -eq "Preferred"
        } |
        Sort-Object InterfaceMetric |
        Select-Object -First 1 -ExpandProperty IPAddress

    return $ip
}

function Test-LocalPort {
    param([int]$Port)

    return Test-NetConnection `
        -ComputerName 127.0.0.1 `
        -Port $Port `
        -InformationLevel Quiet `
        -WarningAction SilentlyContinue
}

$ip = Get-LanIp
if (-not $ip) {
    Write-Host "[ERREUR] IP LAN introuvable" -ForegroundColor Red
    exit 1
}

Write-Host "IP LAN de la machine : $ip" -ForegroundColor Green
Write-Host ""
Write-Host "URLs a partager (autres machines du meme reseau Wi-Fi/LAN) :" -ForegroundColor Cyan

$services = @(
    @{ Name = "API (docs)"; Port = $ApiPort; Path = "/docs" },
    @{ Name = "Frontend"; Port = $FrontendPort; Path = "" },
    @{ Name = "MLflow"; Port = $MlflowPort; Path = "" },
    @{ Name = "Airflow"; Port = $AirflowPort; Path = "" }
)

foreach ($service in $services) {
    $isListening = Test-LocalPort -Port $service.Port
    $state = if ($isListening) { "actif" } else { "hors ligne" }
    $url = "http://$ip`:$($service.Port)$($service.Path)"
    Write-Host (" {0,-11} {1} [{2}]" -f $service.Name, $url, $state)
}

Write-Host ""
Write-Host "Les services Docker sont exposes en 0.0.0.0 via docker compose." -ForegroundColor Yellow
Write-Host "Si une autre machine ne peut pas se connecter, verifier le pare-feu Windows et le reseau Wi-Fi." -ForegroundColor Yellow
