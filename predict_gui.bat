@echo off
:: Batch script to run the FastVLM GUI prediction script

echo Activating fastvlm environment...
call conda activate fastvlm

if errorlevel 1 (
    echo Failed to activate fastvlm environment. Please make sure it exists.
    echo You can create it with: conda create -n fastvlm python=3.10
    pause
    exit /b 1
)

echo Running FastVLM GUI prediction script...
python predict_gui.py

if errorlevel 1 (
    echo Script execution failed.
    pause
    exit /b 1
)

echo Script completed successfully.
pause