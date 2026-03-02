# DataLens (Data Analytics Dashboard)

Single-page analytics platform with React frontend + FastAPI backend.

## Stack
- Frontend: React 18 + TypeScript + TailwindCSS + Recharts
- Backend: FastAPI + pandas + scipy + numpy + statsmodels
- Async Pipeline: Celery + Redis
- AI Summaries: Anthropic Claude (`claude-sonnet-4-20250514`)

## Project Structure
- `frontend/` React app
- `backend/` FastAPI app and Celery workers
- `backend/storage/uploads/` uploaded source files
- `backend/storage/outputs/` charts, cleaned CSV, reports

## Backend Setup
```powershell
cd "e:\Data analytics dashboard\backend"
"e:/Data analytics dashboard/.venv/Scripts/python.exe" -m pip install -r requirements.txt
copy .env.example .env
```

## Redis + Celery + API
1) Start Redis (Docker):
```powershell
docker run --name datalens-redis -p 6379:6379 redis:7
```

2) Start FastAPI:
```powershell
cd "e:\Data analytics dashboard\backend"
"e:/Data analytics dashboard/.venv/Scripts/python.exe" -m uvicorn app.main:app --reload --port 8000
```

3) Start Celery worker:
```powershell
cd "e:\Data analytics dashboard\backend"
"e:/Data analytics dashboard/.venv/Scripts/python.exe" -m celery -A app.workers.celery_app.celery_app worker -l info
```

## Frontend Setup
```powershell
cd "e:\Data analytics dashboard\frontend"
npm install
npm run dev
```

Open `http://localhost:5173`.

## Implemented API Endpoints
- `POST /api/v1/uploads`
- `GET /api/v1/uploads/{upload_id}`
- `POST /api/v1/analysis/{upload_id}/start`
- `GET /api/v1/analysis/{upload_id}/status`
- `GET /api/v1/analysis/{upload_id}/summary`
- `GET /api/v1/analysis/{upload_id}/columns`
- `GET /api/v1/analysis/{upload_id}/columns/{col}/stats`
- `PATCH /api/v1/analysis/{upload_id}/columns/{col}/type`
- `POST /api/v1/analysis/{upload_id}/export/pdf`
- `POST /api/v1/analysis/{upload_id}/share`

Additional export endpoints:
- `GET /api/v1/analysis/{upload_id}/export/cleaned-csv`
- `GET /api/v1/analysis/{upload_id}/export/excel`
