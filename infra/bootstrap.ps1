<#
One-time Azure setup for the RUL service. Run it in PowerShell after `az login`.

It registers the needed resource providers, creates the resource group, and gives GitHub
Actions permission to deploy into that group (and nowhere else) using passwordless OIDC.

Example:
  .\infra\bootstrap.ps1 -GitHubOwner your-username -GitHubRepo rul-service
#>
param(
    [Parameter(Mandatory = $true)][string]$GitHubOwner,
    [Parameter(Mandatory = $true)][string]$GitHubRepo,
    [string]$Location = "centralindia",
    [string]$ResourceGroup = "rg-rul-service",
    [string]$Environment = "production",
    # Repositories created after 15 July 2026 use the immutable subject format (the default).
    # Pass this switch only for an older repository that has not opted in.
    [switch]$LegacySubject
)

$ErrorActionPreference = "Stop"

function Invoke-Az {
    # Run an az command and stop with a clear message if it fails.
    $output = & az @args
    if ($LASTEXITCODE -ne 0) { throw "az $($args -join ' ') failed" }
    return $output
}

$subscriptionId = Invoke-Az account show --query id -o tsv
$tenantId = Invoke-Az account show --query tenantId -o tsv
Write-Host "Subscription: $subscriptionId"

Write-Host "Requesting resource provider registration (it completes in the background)..."
foreach ($ns in @("Microsoft.App", "Microsoft.OperationalInsights", "Microsoft.Consumption")) {
    Invoke-Az provider register --namespace $ns | Out-Null
}

Write-Host "Creating resource group $ResourceGroup in $Location..."
Invoke-Az group create --name $ResourceGroup --location $Location --tags project=rul-service | Out-Null

# GitHub puts numeric owner and repository IDs into the login token of newer repositories.
if ($LegacySubject) {
    $subject = "repo:${GitHubOwner}/${GitHubRepo}:environment:${Environment}"
} else {
    $repo = Invoke-RestMethod -Uri "https://api.github.com/repos/${GitHubOwner}/${GitHubRepo}" `
        -Headers @{ "User-Agent" = "rul-bootstrap" }
    $subject = "repo:${GitHubOwner}@$($repo.owner.id)/${GitHubRepo}@$($repo.id):environment:${Environment}"
}
Write-Host "Trusting GitHub subject: $subject"

Write-Host "Creating the deployment identity..."
$appId = Invoke-Az ad app create --display-name "rul-service-deploy" --query appId -o tsv
Invoke-Az ad sp create --id $appId | Out-Null
$spObjectId = Invoke-Az ad sp show --id $appId --query id -o tsv

$credential = @{
    name      = "github-${Environment}"
    issuer    = "https://token.actions.githubusercontent.com"
    subject   = $subject
    audiences = @("api://AzureADTokenExchange")
} | ConvertTo-Json
$credentialFile = New-TemporaryFile
Set-Content -Path $credentialFile -Value $credential
Invoke-Az ad app federated-credential create --id $appId --parameters "@$credentialFile" | Out-Null
Remove-Item $credentialFile

Write-Host "Granting Contributor on the resource group only..."
$scope = "/subscriptions/${subscriptionId}/resourceGroups/${ResourceGroup}"
Invoke-Az role assignment create --assignee-object-id $spObjectId `
    --assignee-principal-type ServicePrincipal --role Contributor --scope $scope | Out-Null

Write-Host ""
Write-Host "Done. Add these three GitHub repository secrets:"
Write-Host "  AZURE_CLIENT_ID       = $appId"
Write-Host "  AZURE_TENANT_ID       = $tenantId"
Write-Host "  AZURE_SUBSCRIPTION_ID = $subscriptionId"