@echo off
REM experiments\compare.bat - Windows batch to run three experiments and save logs/checkpoints
setlocal enabledelayedexpansion

set ROOT_DIR=%~dp0
set LOG_DIR=%ROOT_DIR%logs
set RES_DIR=%ROOT_DIR%results
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%RES_DIR%" mkdir "%RES_DIR%"

call :run_exp exp_small --epochs 10 --batch-size 8 --seq-len 128 --d-model 128 --num-layers 4 --lr 5e-4 --warmup-steps 1000 --tokenizer bpe --vocab-size 2000 --grad-accum 1
call :run_exp exp_medium --epochs 50 --batch-size 12 --seq-len 256 --d-model 256 --num-layers 6 --lr 5e-4 --warmup-steps 2000 --tokenizer bpe --vocab-size 5000 --grad-accum 2 --gradient-checkpointing
call :run_exp exp_large --epochs 200 --batch-size 16 --seq-len 512 --d-model 512 --num-layers 12 --lr 3e-4 --warmup-steps 4000 --tokenizer bpe --vocab-size 10000 --grad-accum 4

echo All experiments finished. Results are under "%RES_DIR%" and logs under "%LOG_DIR%"
goto :eof

:run_exp
set NAME=%1
shift
set OUTDIR=%RES_DIR%\%NAME%
if not exist "%OUTDIR%" mkdir "%OUTDIR%"
set LOGFILE=%LOG_DIR%\%NAME%.log

echo Running experiment %NAME% ...
python scripts\train.py %* --save-path "%OUTDIR%\%NAME%.pt" > "%LOGFILE%" 2>&1

if exist checkpoints (
  move /Y checkpoints\*.pt "%OUTDIR%" >nul 2>&1
)

echo Finished %NAME% (logs -> %LOGFILE%, results -> %OUTDIR%)
goto :eof
