# Testing LLM Engine Docker Deployment

## Quick Test Summary

You have 3 components running:
- **PostgreSQL** (localhost:5432) - Database
- **Redis** (localhost:6379) - Cache/Queue
- **LLM Engine Gateway** (localhost:5000) - API Server

---

## Test 1: Check Container Health

### View Running Containers

```powershell
docker ps
```

**Expected output:** All 3 containers should show `STATUS: Up`

```
CONTAINER ID   IMAGE          COMMAND                  CREATED        STATUS           PORTS
54c4d547035c   python:3.10    "python -c 'import...'" 5 minutes ago  Up 5 minutes     0.0.0.0:5000->5000/tcp
4b04c526ac75   postgres:14    "docker-entrypoint.s..." 5 minutes ago  Up 5 minutes     0.0.0.0:5432->5432/tcp
ff15c4c1c2d0   redis:7-alpine "docker-entrypoint.s..." 5 minutes ago  Up 5 minutes     0.0.0.0:6379->6379/tcp
```

### Detailed Container Info

```powershell
# Get container details
docker inspect llm-engine-postgres-1 | ConvertFrom-Json | Select-Object -ExpandProperty State
docker inspect llm-engine-redis-1 | ConvertFrom-Json | Select-Object -ExpandProperty State
docker inspect llm-engine-llm-engine-gateway-1 | ConvertFrom-Json | Select-Object -ExpandProperty State
```

---

## Test 2: Check Logs

### Gateway Logs

```powershell
# View last 20 lines
docker logs llm-engine-llm-engine-gateway-1 --tail 20

# Follow logs in real-time
docker logs llm-engine-llm-engine-gateway-1 --follow
```

### Database Logs

```powershell
docker logs llm-engine-postgres-1 --tail 20
```

### Redis Logs

```powershell
docker logs llm-engine-redis-1 --tail 20
```

---

## Test 3: Database Connectivity

### Test PostgreSQL Connection

```powershell
# Connect to database and run a query
docker exec llm-engine-postgres-1 psql -U llm_engine -d llm_engine -c "SELECT 1 as connection_test;"
```

**Expected output:**
```
 connection_test
-----------------
               1
(1 row)
```

### Check Database Size

```powershell
docker exec llm-engine-postgres-1 psql -U llm_engine -d llm_engine -c "\l"
```

**Expected output:** List of databases including `llm_engine`

### Test Database Connectivity from Outside Container

```powershell
# If you have PostgreSQL client installed locally
psql -h localhost -U llm_engine -d llm_engine -c "SELECT version();"

# Or use Docker PostgreSQL client
docker run -it --rm postgres:14 psql -h host.docker.internal -U llm_engine -d llm_engine -c "SELECT 1"
```

---

## Test 4: Redis Connectivity

### Test Redis Connection

```powershell
# Ping Redis
docker exec llm-engine-redis-1 redis-cli ping
```

**Expected output:**
```
PONG
```

### Get Redis Info

```powershell
docker exec llm-engine-redis-1 redis-cli info server
```

**Expected output:** Redis server information including version

### Test Set/Get Operations

```powershell
# Set a key
docker exec llm-engine-redis-1 redis-cli SET test_key "Hello from LLM Engine"

# Get the key
docker exec llm-engine-redis-1 redis-cli GET test_key
```

**Expected output:**
```
Hello from LLM Engine
```

---

## Test 5: API Gateway Connectivity

### Basic Connectivity Test

```powershell
# Test if gateway is responding
$response = Invoke-WebRequest -Uri "http://localhost:5000/" -ErrorAction SilentlyContinue
$response.StatusCode
```

**Expected output:** `200`

### Check Gateway Response

```powershell
# Get the HTML response
Invoke-WebRequest -Uri "http://localhost:5000/" | Select-Object -ExpandProperty Content
```

**Expected output:** Directory listing of container filesystem

### Test API Endpoint (with Basic Auth)

```powershell
# Method 1: Using basic PowerShell (without auth)
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5000/v1/llm/model-endpoints"
    $response.StatusCode
    $response.Content
} catch {
    "Error: $($_.Exception.Message)"
}
```

### Using Python for Better Testing

```powershell
# Create a test script
@"
import requests
import json

# Test 1: Basic connectivity
try:
    response = requests.get('http://localhost:5000/')
    print(f"✓ Gateway responsive: {response.status_code}")
except Exception as e:
    print(f"✗ Gateway error: {e}")

# Test 2: Check API endpoint
try:
    response = requests.get(
        'http://localhost:5000/v1/llm/model-endpoints',
        auth=('test-user-id', '')
    )
    print(f"✓ API endpoint: {response.status_code}")
    print(f"  Response: {response.text}")
except Exception as e:
    print(f"✗ API error: {e}")

# Test 3: Test with JSON
try:
    headers = {'Content-Type': 'application/json'}
    data = {'test': 'data'}
    response = requests.post(
        'http://localhost:5000/test',
        json=data,
        headers=headers
    )
    print(f"✓ POST request: {response.status_code}")
except Exception as e:
    print(f"✗ POST error: {e}")
"@ | Out-File -FilePath "test_gateway.py" -Encoding UTF8

python test_gateway.py
```

---

## Test 6: Docker Network Communication

### Test Service-to-Service Communication

```powershell
# Gateway can reach PostgreSQL?
docker exec llm-engine-llm-engine-gateway-1 /bin/bash -c "curl -v telnet://postgres:5432 2>&1 | head -5"

# Gateway can reach Redis?
docker exec llm-engine-llm-engine-gateway-1 /bin/bash -c "curl -v telnet://redis:6379 2>&1 | head -5"
```

### Check Docker Network

```powershell
# List networks
docker network ls

# Inspect the LLM Engine network
docker network inspect llm-engine_default
```

---

## Test 7: Performance & Resource Usage

### Monitor Container Resources

```powershell
# Real-time resource usage
docker stats llm-engine-postgres-1 llm-engine-redis-1 llm-engine-llm-engine-gateway-1

# One-time snapshot
docker stats --no-stream llm-engine-postgres-1 llm-engine-redis-1 llm-engine-llm-engine-gateway-1
```

**Expected output:**
```
CONTAINER                              CPU %     MEM USAGE / LIMIT
llm-engine-postgres-1                  0.50%     150MiB / 8GiB
llm-engine-redis-1                     0.10%     10MiB / 8GiB
llm-engine-llm-engine-gateway-1        0.01%     30MiB / 8GiB
```

### Check Disk Usage

```powershell
# Docker system usage
docker system df

# Remove unused resources
docker system prune
```

---

## Test 8: Data Persistence

### Test Database Persistence

```powershell
# Create a test table
docker exec llm-engine-postgres-1 psql -U llm_engine -d llm_engine -c "CREATE TABLE IF NOT EXISTS test_table (id SERIAL PRIMARY KEY, name VARCHAR(100));"

# Insert data
docker exec llm-engine-postgres-1 psql -U llm_engine -d llm_engine -c "INSERT INTO test_table (name) VALUES ('Test Data');"

# Query data
docker exec llm-engine-postgres-1 psql -U llm_engine -d llm_engine -c "SELECT * FROM test_table;"

# Restart database container
docker restart llm-engine-postgres-1

# Check if data persists
docker exec llm-engine-postgres-1 psql -U llm_engine -d llm_engine -c "SELECT * FROM test_table;"
```

**Expected:** Data should persist after restart

### Test Redis Persistence

```powershell
# Set a value
docker exec llm-engine-redis-1 redis-cli SET persist_test "This should persist"

# Get value
docker exec llm-engine-redis-1 redis-cli GET persist_test

# Restart Redis
docker restart llm-engine-redis-1

# Check if data persists (may not persist - depends on Redis config)
docker exec llm-engine-redis-1 redis-cli GET persist_test
```

---

## Test 9: Container Health Checks

### Manual Health Check

```powershell
# PostgreSQL health
docker exec llm-engine-postgres-1 pg_isready -U llm_engine

# Redis health
docker exec llm-engine-redis-1 redis-cli ping

# Gateway health
curl http://localhost:5000/ -ErrorAction SilentlyContinue | Select-Object -ExpandProperty StatusCode
```

**Expected outputs:**
```
accepting connections          (PostgreSQL)
PONG                           (Redis)
200                            (Gateway)
```

---

## Test 10: Stress Testing (Optional)

### Generate Load on Gateway

```powershell
# Create load test script
@"
import requests
import concurrent.futures
import time

def make_request(i):
    try:
        response = requests.get('http://localhost:5000/', timeout=5)
        return (i, response.status_code, time.time())
    except Exception as e:
        return (i, 'ERROR', str(e))

# Run 50 requests in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(make_request, i) for i in range(50)]
    results = [f.result() for f in concurrent.futures.as_completed(futures)]

# Print results
success = sum(1 for _, code, _ in results if code == 200)
failed = len(results) - success

print(f"✓ Successful: {success}/50")
print(f"✗ Failed: {failed}/50")
"@ | Out-File -FilePath "load_test.py" -Encoding UTF8

python load_test.py
```

### Monitor During Load Test

```powershell
# In another terminal, run:
docker stats llm-engine-llm-engine-gateway-1 --no-stream
```

---

## Comprehensive Test Script

Save this as `full_test.ps1` and run it:

```powershell
# ============================================================================
# LLM Engine Docker Deployment - Comprehensive Test Suite
# ============================================================================

Write-Host "=== LLM Engine Docker Deployment Tests ===" -ForegroundColor Cyan

# Test 1: Container Status
Write-Host "`n[Test 1] Container Status" -ForegroundColor Yellow
$containers = docker ps --filter "name=llm-engine" --format "{{.Names}}`t{{.Status}}"
if ($containers.Count -ge 3) {
    Write-Host "✓ All 3 containers running" -ForegroundColor Green
    $containers | ForEach-Object { Write-Host "  $_" }
} else {
    Write-Host "✗ Not all containers running" -ForegroundColor Red
}

# Test 2: PostgreSQL
Write-Host "`n[Test 2] PostgreSQL Connection" -ForegroundColor Yellow
try {
    $result = docker exec llm-engine-postgres-1 psql -U llm_engine -d llm_engine -c "SELECT 1" 2>&1
    if ($result -match "1 row") {
        Write-Host "✓ PostgreSQL is responding" -ForegroundColor Green
    } else {
        Write-Host "✗ PostgreSQL response unexpected" -ForegroundColor Red
    }
} catch {
    Write-Host "✗ PostgreSQL connection failed: $_" -ForegroundColor Red
}

# Test 3: Redis
Write-Host "`n[Test 3] Redis Connection" -ForegroundColor Yellow
try {
    $result = docker exec llm-engine-redis-1 redis-cli ping 2>&1
    if ($result -eq "PONG") {
        Write-Host "✓ Redis is responding" -ForegroundColor Green
    } else {
        Write-Host "✗ Redis response unexpected" -ForegroundColor Red
    }
} catch {
    Write-Host "✗ Redis connection failed: $_" -ForegroundColor Red
}

# Test 4: Gateway
Write-Host "`n[Test 4] Gateway Connectivity" -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5000/" -TimeoutSec 5 -ErrorAction SilentlyContinue
    if ($response.StatusCode -eq 200) {
        Write-Host "✓ Gateway is responding (HTTP 200)" -ForegroundColor Green
    } else {
        Write-Host "✗ Gateway returned HTTP $($response.StatusCode)" -ForegroundColor Red
    }
} catch {
    Write-Host "✗ Gateway connection failed: $_" -ForegroundColor Red
}

# Test 5: Port Availability
Write-Host "`n[Test 5] Port Availability" -ForegroundColor Yellow
$ports = @(5000, 5432, 6379)
foreach ($port in $ports) {
    try {
        $tcpClient = New-Object System.Net.Sockets.TcpClient
        $tcpClient.Connect("localhost", $port)
        if ($tcpClient.Connected) {
            Write-Host "✓ Port $port is open and listening" -ForegroundColor Green
        }
        $tcpClient.Close()
    } catch {
        Write-Host "✗ Port $port is not accessible" -ForegroundColor Red
    }
}

# Test 6: Resource Usage
Write-Host "`n[Test 6] Resource Usage" -ForegroundColor Yellow
$stats = docker stats --no-stream llm-engine-postgres-1 llm-engine-redis-1 llm-engine-llm-engine-gateway-1
Write-Host $stats

# Test 7: Docker Logs Summary
Write-Host "`n[Test 7] Recent Errors in Logs" -ForegroundColor Yellow
$errors_pg = docker logs llm-engine-postgres-1 --tail 50 | Select-String -Pattern "ERROR|FATAL" -SimpleMatch
$errors_redis = docker logs llm-engine-redis-1 --tail 50 | Select-String -Pattern "Error|error" -SimpleMatch
$errors_gateway = docker logs llm-engine-llm-engine-gateway-1 --tail 50 | Select-String -Pattern "ERROR|Exception" -SimpleMatch

if (-not $errors_pg) { Write-Host "✓ No errors in PostgreSQL logs" -ForegroundColor Green }
else { Write-Host "✗ Found errors in PostgreSQL logs" -ForegroundColor Red; $errors_pg }

if (-not $errors_redis) { Write-Host "✓ No errors in Redis logs" -ForegroundColor Green }
else { Write-Host "✗ Found errors in Redis logs" -ForegroundColor Red; $errors_redis }

if (-not $errors_gateway) { Write-Host "✓ No errors in Gateway logs" -ForegroundColor Green }
else { Write-Host "✗ Found errors in Gateway logs" -ForegroundColor Red; $errors_gateway }

Write-Host "`n=== Test Suite Complete ===" -ForegroundColor Cyan
```

---

## Summary: What Should Pass

✅ **All 3 containers running**
✅ **PostgreSQL accepts connections**
✅ **Redis responds with PONG**
✅ **Gateway returns HTTP 200**
✅ **All ports (5000, 5432, 6379) are open**
✅ **No FATAL errors in logs**
✅ **Memory usage < 500MB total**

---

## If Tests Fail

### Container Won't Start

```powershell
# Check logs
docker logs llm-engine-postgres-1
docker logs llm-engine-redis-1
docker logs llm-engine-llm-engine-gateway-1

# Restart container
docker restart llm-engine-postgres-1

# Remove and recreate
docker rm llm-engine-postgres-1
python engine_controller.py --mode docker --action deploy
```

### Port Already in Use

```powershell
# Find what's using the port
netstat -ano | findstr ":5000"
netstat -ano | findstr ":5432"
netstat -ano | findstr ":6379"

# Kill the process (replace PID)
taskkill /PID <PID> /F

# Try again
python engine_controller.py --mode docker --action cleanup
python engine_controller.py --mode docker --action deploy
```

### Network Issues

```powershell
# Check Docker network
docker network ls
docker network inspect llm-engine_default

# Check DNS resolution
docker exec llm-engine-llm-engine-gateway-1 nslookup postgres
docker exec llm-engine-llm-engine-gateway-1 nslookup redis
```

### Database Connection Issues

```powershell
# Check PostgreSQL is accepting connections
docker logs llm-engine-postgres-1 | Select-String "ready to accept"

# Manually test connection
docker run -it --rm postgres:14 psql -h host.docker.internal -U llm_engine -d llm_engine -c "SELECT 1"
```

---

## Next Steps After Testing

1. **If all tests pass**: Explore the API, set up models, test inference
2. **If some tests fail**: Check logs and troubleshoot above
3. **Ready for Minikube**: `python engine_controller.py --mode local --action deploy`
4. **Ready for AWS**: See `EXPERT_ASSESSMENT.md` and provision AWS resources

