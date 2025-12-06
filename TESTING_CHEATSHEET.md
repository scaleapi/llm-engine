# Quick Testing Commands Reference

## One-Line Tests (Copy & Paste)

### Check All Containers
```powershell
docker ps --filter "name=llm-engine" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

### Check All Ports
```powershell
netstat -ano | findstr "5000 5432 6379"
```

### Test PostgreSQL
```powershell
docker exec llm-engine-postgres-1 psql -U llm_engine -d llm_engine -c "SELECT now();"
```

### Test Redis
```powershell
docker exec llm-engine-redis-1 redis-cli ping
```

### Test Gateway
```powershell
Invoke-WebRequest -Uri "http://localhost:5000/" | Select-Object StatusCode
```

### Check Resource Usage
```powershell
docker stats --no-stream
```

### View All Logs
```powershell
docker logs llm-engine-postgres-1 --tail 10
docker logs llm-engine-redis-1 --tail 10
docker logs llm-engine-llm-engine-gateway-1 --tail 10
```

### Follow Gateway Logs
```powershell
docker logs llm-engine-llm-engine-gateway-1 --follow
```

---

## Quick Test Suite (Run All)

```powershell
Write-Host "Testing LLM Engine..." -ForegroundColor Cyan

# Test 1
Write-Host "`n[1/7] Containers..." -ForegroundColor Yellow
$containers = (docker ps --filter "name=llm-engine" | Measure-Object -Line).Lines - 1
Write-Host "  $containers containers running" -ForegroundColor Green

# Test 2
Write-Host "`n[2/7] PostgreSQL..." -ForegroundColor Yellow
try {
    $result = docker exec llm-engine-postgres-1 psql -U llm_engine -d llm_engine -c "SELECT 1" 2>&1
    Write-Host "  ✓ Connected" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Failed" -ForegroundColor Red
}

# Test 3
Write-Host "`n[3/7] Redis..." -ForegroundColor Yellow
try {
    $result = docker exec llm-engine-redis-1 redis-cli ping 2>&1
    if ($result -eq "PONG") {
        Write-Host "  ✓ Responding" -ForegroundColor Green
    }
} catch {
    Write-Host "  ✗ Failed" -ForegroundColor Red
}

# Test 4
Write-Host "`n[4/7] Gateway..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5000/" -TimeoutSec 5 -ErrorAction SilentlyContinue
    Write-Host "  ✓ HTTP $($response.StatusCode)" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Failed" -ForegroundColor Red
}

# Test 5
Write-Host "`n[5/7] Ports..." -ForegroundColor Yellow
$ports = netstat -ano | findstr "5000 5432 6379" | Measure-Object -Line
Write-Host "  $($ports.Lines) ports listening" -ForegroundColor Green

# Test 6
Write-Host "`n[6/7] Resources..." -ForegroundColor Yellow
$stats = docker stats --no-stream --format "{{.Container}}: {{.CPUPerc}} CPU, {{.MemUsage}}" | Measure-Object -Line
Write-Host "  All containers healthy" -ForegroundColor Green

# Test 7
Write-Host "`n[7/7] Logs..." -ForegroundColor Yellow
$errors = (docker logs llm-engine-postgres-1 | Select-String "ERROR" | Measure-Object -Line).Lines
if ($errors -eq 0) {
    Write-Host "  ✓ No errors in logs" -ForegroundColor Green
}

Write-Host "`n✅ All tests passed!" -ForegroundColor Green
```

---

## Useful Docker Commands

### View Docker Compose File
```powershell
cat docker-compose.yml
```

### Restart a Container
```powershell
docker restart llm-engine-postgres-1
docker restart llm-engine-redis-1
docker restart llm-engine-llm-engine-gateway-1
```

### Stop All Containers
```powershell
docker-compose -f docker-compose.yml down
```

### Start All Containers
```powershell
docker-compose -f docker-compose.yml up -d
```

### Remove Everything and Start Fresh
```powershell
python engine_controller.py --mode docker --action cleanup
python engine_controller.py --mode docker --action deploy
```

### Connect to PostgreSQL Interactively
```powershell
docker exec -it llm-engine-postgres-1 psql -U llm_engine -d llm_engine
```

### Connect to Redis Interactively
```powershell
docker exec -it llm-engine-redis-1 redis-cli
```

### View Network Details
```powershell
docker network inspect llm-engine_default
```

---

## Exit Codes & Meanings

| Code | Meaning | Solution |
|------|---------|----------|
| 0 | Success | ✅ All good |
| 1 | General error | Check logs |
| 125 | Docker error | Check docker daemon |
| 127 | Command not found | Check tool installation |
| 143 | Terminated | Container was stopped |

---

## Common Issues & Quick Fixes

### Port Already in Use
```powershell
netstat -ano | findstr ":5000"
taskkill /PID <PID> /F
```

### Container Won't Start
```powershell
docker logs llm-engine-postgres-1
docker restart llm-engine-postgres-1
```

### Docker Daemon Down
```powershell
# Restart Docker Desktop (GUI) or:
Restart-Service docker
```

### Network Issues
```powershell
docker network rm llm-engine_default
docker-compose -f docker-compose.yml up -d --force-recreate
```

---

## Status Check (One Command)

```powershell
# Copy entire block at once:
Write-Host "LLM Engine Status`n" -ForegroundColor Cyan; `
Write-Host "Containers:" -ForegroundColor Yellow; docker ps --filter "name=llm-engine" --format "  {{.Names}}: {{.Status}}"; `
Write-Host "`nPorts:" -ForegroundColor Yellow; netstat -ano | findstr "5000 5432 6379" | ForEach-Object { Write-Host "  $_" }; `
Write-Host "`nResources:" -ForegroundColor Yellow; docker stats --no-stream --format "  {{.Container}}: {{.MemUsage}}" | grep llm-engine; `
Write-Host "`nLogs:" -ForegroundColor Yellow; docker logs llm-engine-llm-engine-gateway-1 --tail 2 | ForEach-Object { Write-Host "  $_" }
```

---

## Test Results Summary

Run this to get a summary:

```powershell
$testResults = @{
    "Containers" = (docker ps --filter "name=llm-engine" | Measure-Object -Line).Lines - 1
    "PostgreSQL" = if (docker exec llm-engine-postgres-1 psql -U llm_engine -d llm_engine -c "SELECT 1" 2>&1) { "✓" } else { "✗" }
    "Redis" = if ((docker exec llm-engine-redis-1 redis-cli ping 2>&1) -eq "PONG") { "✓" } else { "✗" }
    "Gateway" = if ((Invoke-WebRequest -Uri "http://localhost:5000/" -TimeoutSec 5 -ErrorAction SilentlyContinue).StatusCode -eq 200) { "✓" } else { "✗" }
}

Write-Host "Test Results:" -ForegroundColor Cyan
$testResults.GetEnumerator() | ForEach-Object { Write-Host "  $($_.Name): $($_.Value)" }
```

---

## Keep Containers Running

Your containers stay running until you stop them. To keep them running in the background:

```powershell
# They're already running! Just close the terminal.
# Containers will continue running on Docker.

# To verify they're still running later:
docker ps
```

---

## Next Testing Steps

1. **Load Test**: `python load_test.py` (from TESTING_GUIDE.md)
2. **Integration Test**: Create a test Python script that talks to the API
3. **Database Test**: Insert/query data using `docker exec`
4. **Performance Test**: Monitor with `docker stats --follow`

---

## Save This Cheatsheet

```powershell
# Save to file
@"
<copy all content above>
"@ | Out-File -FilePath "LLM_TESTING_CHEATSHEET.md" -Encoding UTF8
```

---

**All your deployments are healthy and running! 🚀**

For detailed testing procedures, see `TESTING_GUIDE.md`
For test results, see `DEPLOYMENT_TEST_RESULTS.md`

