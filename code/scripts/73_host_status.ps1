# 73_host_status.ps1 -- one-shot machine status for the 5-minute monitor. ASCII only (PS5 reads ANSI).
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$o = Get-CimInstance Win32_OperatingSystem
'NOW=' + (Get-Date -Format 'MM-dd HH:mm:ss') + '  BOOT=' + $o.LastBootUpTime.ToString('MM-dd HH:mm')
'RAM_FREE={0:N1}GB  COMMIT={1:N1}/{2:N1}GB ({3:P0})' -f ($o.FreePhysicalMemory/1MB), (($o.TotalVirtualMemorySize-$o.FreeVirtualMemory)/1MB), ($o.TotalVirtualMemorySize/1MB), (($o.TotalVirtualMemorySize-$o.FreeVirtualMemory)/$o.TotalVirtualMemorySize)
$pf = Get-CimInstance Win32_PageFileUsage | Select-Object -First 1
'PAGEFILE alloc={0}MB used={1}MB peak={2}MB' -f $pf.AllocatedBaseSize, $pf.CurrentUsage, $pf.PeakUsage
$g = (nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm,display_active --format=csv,noheader,nounits) -split ',\s*'
'GPU util={0}%  vram={1}MiB  temp={2}C  power={3}W  sm={4}MHz  display={5}' -f $g[0],$g[1],$g[2],$g[3],$g[4],$g[5]
$roles = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ForEach-Object {
    $pr = Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue
    $r = if ($_.CommandLine -match '70_chain_fusai.py --tag (\w+)') { 'chain' + $matches[1] }
         elseif ($_.CommandLine -match 'multiprocessing') { 'worker' }
         elseif ($_.CommandLine -match '28_finetune') { 'train' }
         elseif ($_.CommandLine -match '58_test_viewpool') { 'pool' }
         elseif ($_.CommandLine -match '59_recipe|30_pseudo') { 'pack' }
         else { 'other' }
    '{0}({1:N1}G)' -f $r, ($pr.PrivateMemorySize64/1GB)
}
'PY[' + @($roles).Count + ']=' + ($roles -join ' ')
$bg = Get-Process douyin,huyapc,HuyaClient,msedge,Widgets,kwallpaperplayer,Doubao -ErrorAction SilentlyContinue |
      Group-Object ProcessName | ForEach-Object { $_.Name + 'x' + $_.Count }
'BG_APPS=' + ($(if ($bg) { $bg -join ',' } else { 'none' }))
