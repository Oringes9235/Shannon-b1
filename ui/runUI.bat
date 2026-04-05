@echo off
setlocal enabledelayedexpansion
title Shannon-b1 Web UI

echo ==========================================
echo    Shannon-b1 Web UI Launcher
echo ==========================================
echo.

REM Resolve script directory
set SCRIPT_DIR=%~dp0

REM Check Python
python --version > nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found
    pause
    exit /b 1
)

REM Check Node.js
node --version > nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found
    pause
    exit /b 1
)

:: Backend: install deps and start (in background)
echo [1/5] Installing backend dependencies...
pushd "%SCRIPT_DIR%server"
if exist requirements.txt (
    python -m pip install -r requirements.txt -q
) else (
    python -m pip install fastapi uvicorn websockets python-multipart aiofiles -q
)
echo        Done

echo [2/5] Starting backend server (uvicorn) in a new terminal...
REM start uvicorn in a new visible window so user can see logs and stop it manually
start "Shannon-Backend" cmd /k "cd /d "%SCRIPT_DIR%server" && python -m uvicorn app:app --host 0.0.0.0 --port 8000"

echo Waiting for backend to become available (checking http://localhost:8000/api/status)...
set /a WAIT_SECS=0
set /a MAX_WAIT=60
:wait_loop
    curl --silent --fail http://localhost:8000/api/status > nul 2>&1
    if %errorlevel%==0 (
        echo Backend is up (after %WAIT_SECS% seconds)
        goto backend_ready
    )
    if %WAIT_SECS% geq %MAX_WAIT% (
        echo Backend did not start within %MAX_WAIT% seconds. Check UI server window for errors.
        goto backend_failed
    )
    timeout /t 1 /nobreak > nul
    set /a WAIT_SECS+=1
    goto wait_loop

:backend_ready
echo [3/5] Backend ready. Proceeding to frontend.

:: Frontend: ask user which mode to run (dev or preview)
pushd "%SCRIPT_DIR%client"
if not exist "node_modules" (
    echo        Installing frontend dependencies (first time may take a few minutes)...
    call npm install
) else (
    echo        Frontend dependencies present
)

set /p FRONT_MODE=Choose frontend mode: [dev/preview] (default: preview): 
if "%FRONT_MODE%"=="" set FRONT_MODE=preview

if /I "%FRONT_MODE%"=="dev" (
    echo Starting frontend in DEV mode in a new terminal (visible)...
    start "Shannon-Frontend (dev)" cmd /k "cd /d "%SCRIPT_DIR%client" && npm run dev"
) else (
    echo Building frontend for preview (vite build)...
    call npm run build
    echo        Build finished
    echo Starting frontend preview server in a new terminal (visible)...
    start "Shannon-Frontend (preview)" cmd /k "cd /d "%SCRIPT_DIR%client" && npm run preview -- --port 5173"
)
popd

if /I "%FRONT_MODE%"=="preview" (
    echo Opening http://localhost:5173
    start http://localhost:5173
)

echo.
echo ==========================================
echo    SUCCESS! Services started (background)
echo    Frontend: http://localhost:5173
echo    Backend:  http://localhost:8000
echo    API Docs: http://localhost:8000/docs
echo ==========================================

pause
