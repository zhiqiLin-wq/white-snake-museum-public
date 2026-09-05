@echo off
REM B-138: 本地开发环境一键启动脚本
echo ============================================
echo   White Snake Museum - Dev Startup
echo ============================================

REM 启动 Node 代理 (port 3000)
echo [1/3] Starting Node proxy server...
start "Node Proxy" cmd /c "cd /d %~dp0..\server && npm run dev"

REM 等待 Node 启动
timeout /t 3 /nobreak >nul

REM 启动 Python Agent (port 8000)
echo [2/3] Starting Python Agent...
start "Python Agent" cmd /c "cd /d %~dp0..\agent && venv\Scripts\python -m server.main"

REM 等待 Agent 启动
timeout /t 5 /nobreak >nul

REM 检查服务健康状态
echo [3/3] Checking service health...
curl -s http://localhost:3000/api/agent/health
echo.
curl -s http://localhost:8000/health
echo.

echo ============================================
echo   All services started!
echo   Node Proxy:  http://localhost:3000
echo   Python Agent: http://localhost:8000
echo   API Docs:     http://localhost:8000/docs
echo ============================================
pause
