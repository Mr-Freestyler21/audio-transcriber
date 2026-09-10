@echo off
REM Transcribe from the command line. You can also DRAG an audio file onto this .bat.
cd /d "%~dp0"
if "%~1"=="" (
  echo.
  echo   Usage: drag an audio file onto this file, or run:
  echo          transcribe.bat myaudio.mp3
  echo.
  pause
  exit /b 0
)
".venv\Scripts\python.exe" transcribe.py %*
pause
