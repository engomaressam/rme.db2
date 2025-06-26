$batPath = "C:\\reem\\02.new.reem\\run_email_script_with_log.bat"
$action = New-ScheduledTaskAction -Execute $batPath
$trigger = New-ScheduledTaskTrigger -Daily -At 3PM
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive
$taskName = "ReemEmailSummaryDaily"

# Remove the task if it already exists
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal
Write-Host "Scheduled task '$taskName' created to run daily at 3 PM." 