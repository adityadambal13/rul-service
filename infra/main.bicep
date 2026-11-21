// RUL service on Azure Container Apps (Consumption plan, scales to zero).
targetScope = 'resourceGroup'

@description('Azure region. Defaults to the resource group location (Central India = Pune).')
param location string = resourceGroup().location

@description('Base name for the resources.')
param name string = 'rul-service'

@description('Full container image reference, e.g. ghcr.io/owner/rul-service:abc123')
param image string

@description('Minimum replicas. 0 = scale to zero (free, but the first request is slow).')
@minValue(0)
param minReplicas int = 0

@description('Maximum replicas. Keep at 1 to cap cost.')
@minValue(1)
param maxReplicas int = 1

@description('API key required by /v1/predict. Leave empty to make the API open.')
@secure()
param apiKey string = ''

var tags = {
  project: 'rul-service'
}

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${name}-logs'
  location: location
  tags: tags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
    workspaceCapping: {
      dailyQuotaGb: json('0.5') // hard cap on log ingestion, a cost guardrail
    }
  }
}

resource env 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${name}-env'
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: name
  location: location
  tags: tags
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
        allowInsecure: false
      }
      secrets: empty(apiKey) ? [] : [
        {
          name: 'api-key'
          value: apiKey
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: image
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: empty(apiKey) ? [] : [
            {
              name: 'RUL_API_KEY'
              secretRef: 'api-key'
            }
          ]
          probes: [
            {
              type: 'Startup'
              httpGet: {
                path: '/health'
                port: 8000
              }
              periodSeconds: 3
              failureThreshold: 40
            }
            {
              type: 'Liveness'
              httpGet: {
                path: '/health'
                port: 8000
              }
              periodSeconds: 30
            }
            {
              type: 'Readiness'
              httpGet: {
                path: '/ready'
                port: 8000
              }
              periodSeconds: 10
              failureThreshold: 6
            }
          ]
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: maxReplicas
        rules: [
          {
            name: 'http-concurrency'
            http: {
              metadata: {
                concurrentRequests: '20'
              }
            }
          }
        ]
      }
    }
  }
}

output appUrl string = 'https://${app.properties.configuration.ingress.fqdn}'
