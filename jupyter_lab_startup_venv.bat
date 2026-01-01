@echo off

@REM Add double quotes around pathnames if there are spaces in them.

@REM Start Jupyter Lab
echo Starting Jupyter Lab ... please stand by ...
title Jupyter Lab Startup Script
set PROJECT_DIR=C:\Courses\2025\Udemy\YOLO-11-12
call %PROJECT_DIR%\.venv\Scripts\activate.bat
call %PROJECT_DIR%\.venv\Scripts\jupyter-lab.exe %PROJECT_DIR%

pause
