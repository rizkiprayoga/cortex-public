' ============================================================================
'  autostart_hidden.vbs -- Invisible launcher for Task Scheduler autostart.
'
'  Task Scheduler runs this via wscript.exe, which has no console. It in turn
'  launches start_trading_bot.bat with ShowWindow=0 (hidden). The batch file
'  then calls launch.ps1 with -WindowStyle Hidden, so nothing visible ever
'  appears on screen -- no cmd flash, no PS window, nothing to close.
'
'  Third arg to Run is TRUE (wait-for-completion). Critical: without this
'  wscript exits in ~1s and Task Scheduler treats the task as "done", which
'  breaks the MultipleInstances=IgnoreNew guarantee and lets subsequent
'  triggers (logon events, manual fires) spawn duplicate bots. With TRUE,
'  wscript stays alive for the whole bot lifetime so TS sees the task as
'  "running" and blocks duplicate starts.
'
'  All output is still captured in data\logs\autostart.log and trading_bot.log.
' ============================================================================

Set sh = CreateObject("WScript.Shell")
sh.Run """g:\AI_Trading_Bot\Cortex\scripts\start_trading_bot.bat""", 0, True
