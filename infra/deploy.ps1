<#
Deploy a published image to Azure Container Apps from your own machine.

Usage:
  .\infra\deploy.ps1 -GitHubOwner your-username
  .\infra\deploy.ps1 -GitHubOwner your-username -Sha <commit sha>

Run `az login` first. Set the RUL_API_KEY environment variable (or pass -ApiKey)
to protect the API; leave it empty for an open API.
#>
param(
    [Parameter(Mandatory = $true)][string]$GitHubOwner,
    [string]$GitHubRepo = "rul-service",
    [string]$Sha = "",
    [string]$ResourceGroup = "rg-rul-service",
    [string]$ApiKey = $env:RUL_API_KEY
)

$ErrorActionPreference = "Stop"

if (-not $Sha) {
    git fetch origin main | Out-Null
    $Sha = (git rev-parse origin/main).Trim()
}
$image = "ghcr.io/$($GitHubOwner.ToLower())/$($GitHubRepo.ToLower()):$Sha"
Write-Host "Deploying $image"

$url = az deployment group create `
    --resource-group $ResourceGroup `
    --template-file infra/main.bicep `
    --parameters image=$image apiKey=$ApiKey `
    --query properties.outputs.appUrl.value -o tsv
if ($LASTEXITCODE -ne 0) { throw "The deployment failed." }
Write-Host "Deployed to $url"

Write-Host "Waiting for the service to start (the first start can take a minute or two)..."
$ready = $false
for ($i = 1; $i -le 60; $i++) {
    try {
        Invoke-RestMethod -Uri "$url/ready" | Out-Null
        $ready = $true
        break
    } catch {
        Start-Sleep -Seconds 5
    }
}
if (-not $ready) { throw "The service did not become ready in time. Check the logs." }

$headers = @{}
if ($ApiKey) { $headers["X-API-Key"] = $ApiKey }
$body = Get-Content deploy/sample_request.json -Raw
$result = Invoke-RestMethod -Method Post -Uri "$url/v1/predict" `
    -ContentType "application/json" -Headers $headers -Body $body
Write-Host "Prediction from the live service:"
$result | ConvertTo-Json
Write-Host "Live docs: $url/docs"