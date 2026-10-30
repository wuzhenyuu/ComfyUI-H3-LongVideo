@echo off
setlocal
set "requirements_txt=%~dp0requirements.txt"
for %%I in ("%~dp0..\..") do set "comfy_root=%%~fI"
set "python_exec=%~1"

if not defined python_exec if exist "%comfy_root%\venv\Scripts\python.exe" set "python_exec=%comfy_root%\venv\Scripts\python.exe"
if not defined python_exec if exist "%comfy_root%\.venv\Scripts\python.exe" set "python_exec=%comfy_root%\.venv\Scripts\python.exe"
if not defined python_exec if exist "%comfy_root%\..\python_embeded\python.exe" set "python_exec=%comfy_root%\..\python_embeded\python.exe"
if not defined python_exec goto missing_python
if not exist "%python_exec%" goto missing_python

echo Installing ComfyUI-H3-LongVideo dependencies.
echo Using Python: "%python_exec%"
"%python_exec%" -m pip install -r "%requirements_txt%"
set "install_result=%errorlevel%"
if not "%install_result%"=="0" echo Installation failed. Review the error above.
if "%install_result%"=="0" echo Dependencies installed. Model downloads are separate; see MODEL_DOWNLOADS.md.
pause
exit /b %install_result%

:missing_python
echo ComfyUI Python was not found. No global Python or pip was used.
echo Run this installer with the full path of your ComfyUI Python as its first argument.
pause
exit /b 1
