# Increase timeout and stability settings
$env:POWERSHELL_TELEMETRY_OPTOUT = 1
$env:PSES_TIMEOUT = 300000  # 5 minutes timeout
$env:PSES_FORCE_NO_DEBUG = 1

# Disable background jobs and host process
$env:PSES_DISABLE_BACKGROUND_JOBS = 1
$env:PSES_DISABLE_HOSTING_PROCESS = 1

# Improve performance
$PSDefaultParameterValues['*:Encoding'] = 'utf8'
$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'

# Disable update notifications
$env:POWERSHELL_UPDATECHECK = 'Off'

# Enable better error handling
$ErrorView = 'CategoryView'

# Optimize memory usage
[System.GC]::Collect()
[System.GC]::WaitForPendingFinalizers()

# Improve module loading
$env:PSModulePath = [System.Environment]::GetEnvironmentVariable("PSModulePath", "Machine")

# Set execution policy for current session
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force

# Disable some intensive features
$env:PSES_DISABLE_DIAGNOSTICS = 1
$env:PSES_DISABLE_SNIPPETS = 1

# Increase memory limits
$env:PSModuleAnalysisCachePath = "$env:TEMP\PSModuleAnalysisCache"
$env:POWERSHELL_TELEMETRY_OPTOUT = 1

# Set stable encoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8 