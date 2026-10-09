@echo off
rem One click: virtual environment, packages, data, tests, notebooks.
cd /d "%~dp0.."
set PYTHONUTF8=1
if exist "scripts\local.bat" call "scripts\local.bat"
if not exist "logs" mkdir "logs"

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  python -m venv .venv
  if errorlevel 1 (echo [ERROR] Python not found. & pause & exit /b 1)
)
set "PY=.venv\Scripts\python.exe"
echo Installing packages...
"%PY%" -m pip install -q --upgrade pip
"%PY%" -m pip install -q -r requirements.txt
if errorlevel 1 (echo [ERROR] pip failed. & pause & exit /b 1)

echo Downloading data...
"%PY%" scripts\get_data.py
if errorlevel 1 (echo [ERROR] Data download failed. & pause & exit /b 1)

echo Running tests...
"%PY%" -m pytest -q > logs\tests.txt 2>&1
set "TESTS_RC=%errorlevel%"
type logs\tests.txt
if not "%TESTS_RC%"=="0" (echo [ERROR] Tests failed, see logs\tests.txt & pause & exit /b 1)

echo Building notebooks (about a minute)...
"%PY%" scripts\build_notebooks.py > logs\notebooks.txt 2>&1
set "NB_RC=%errorlevel%"
type logs\notebooks.txt
if not "%NB_RC%"=="0" (echo [ERROR] Notebook build failed, see logs\notebooks.txt & pause & exit /b 1)
echo.
echo Done. Notebooks are in the notebooks folder, charts in reports\figures.
pause
