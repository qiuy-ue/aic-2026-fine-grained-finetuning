# launch_A1r_A2.ps1 -- resume A1 from its last saved epoch, then queue A2.
# Double-click START_A1_resume_A2.bat to run this. ASCII only on purpose:
# Windows PowerShell 5 reads BOM-less .ps1 files in the ANSI code page.

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$py   = 'python'
$a1dir = 'runs\20260924_190942_v2_448_cos_ct15_clip0_dp020_re25_ep12'
Set-Location $root

Write-Host '=================================================================='
Write-Host ' AIC fusai: resume A1 (co-teaching + pseudo teacher), queue A2'
Write-Host '=================================================================='

# 1. refuse to double-launch: two copies of A1 on one 8GB GPU would both die
$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*70_chain_fusai*' -or $_.CommandLine -like '*28_finetune_v2*' }
if ($running) {
    Write-Host ''
    Write-Host '[STOP] Training is already running. Not starting a second copy:' -ForegroundColor Yellow
    $running | ForEach-Object { Write-Host ('   PID {0}  {1}' -f $_.ProcessId, $_.CommandLine.Substring(0, [Math]::Min(120, $_.CommandLine.Length))) }
    Write-Host ''
    Write-Host 'If you really want to restart, first run:  Get-Process python | Stop-Process'
    exit 1
}
if (-not (Test-Path (Join-Path $root "$a1dir\ckpt_ep01.pt"))) {
    Write-Host "[STOP] $a1dir\ckpt_ep01.pt not found - nothing to resume from." -ForegroundColor Red
    exit 1
}

# 2. memory check: this laptop has 15.8 GB RAM; background apps already used ~15 GB commit on 09-24
$os = Get-CimInstance Win32_OperatingSystem
$freeGB = $os.FreePhysicalMemory / 1MB
Write-Host ('RAM free: {0:N1} GB of {1:N1} GB' -f $freeGB, ($os.TotalVisibleMemorySize / 1MB))
if ($freeGB -lt 6) {
    Write-Host '[WARN] Less than 6 GB free. Close Douyin / Huya / live wallpaper / Doubao / Edge / WeChat first,' -ForegroundColor Yellow
    Write-Host '       or the machine may freeze again like 09-24 21:03.' -ForegroundColor Yellow
}

# 3. launch detached + hidden: closing this window will NOT stop training
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$a = Start-Process -FilePath $py -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -ArgumentList "scripts\70_chain_fusai.py --tag A1 --co-teaching --make-pseudo --resume-train $a1dir" `
    -RedirectStandardOutput "$root\runs\chain_A1r.log" -RedirectStandardError "$root\runs\chain_A1r.err"
$b = Start-Process -FilePath $py -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -ArgumentList 'scripts\70_chain_fusai.py --tag A2 --co-teaching --pseudo cache\pseudo_A1.npz --wait-for runs\_chain_A1_DONE.txt' `
    -RedirectStandardOutput "$root\runs\chain_A2.log" -RedirectStandardError "$root\runs\chain_A2.err"
Write-Host ''
Write-Host ('[OK] A1 resume started, PID {0}' -f $a.Id) -ForegroundColor Green
Write-Host ('[OK] A2 queued (waits for A1 to finish), PID {0}' -f $b.Id) -ForegroundColor Green
Write-Host ''
Write-Host 'Training runs hidden in the background. You can close this window any time.'
Write-Host 'Do NOT: sleep the laptop, close the lid, or kill python.exe in Task Manager.'
Write-Host ''
Write-Host '--- live log (Ctrl+C or close window to stop watching; training keeps going) ---'
Start-Sleep -Seconds 3
Get-Content "$root\runs\chain_A1r.log" -Wait -Tail 40 -Encoding UTF8
