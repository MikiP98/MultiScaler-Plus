@cd src

start "Backend" cmd /c "uvicorn server:app --workers=8"

@cd MultiScaler-Plus_Web_Server
start "Frontend" cmd /c "npm run dev -- --open --host"

@PAUSE