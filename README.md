# TubeMind

# Tech Stack

1) [FastHTML](https://fastht.ml/docs/): an HTMX based python web framework
2) [MonsterUI](https://monsterui.answer.ai/): a UI library for FastHTML
3) [LightRAG](https://github.com/HKUDS/LightRAG): graphrag implementation

# Run The Server

1) Install dependencies with `uv sync`
2) Create a `.env` file and populate the required values, using `.env.example` as the template
3) Start the app with `uv run python -m tubemind`
4) Open `http://localhost:5001`

# Railway Deploy

For a coursework deploy, the easiest path is Railway with a mounted volume and demo auth enabled.

1) Push this repo to GitHub
2) In Railway, create a new service from the repo
3) Add a volume and mount it at `/app/.local`
4) Add these service variables:

```env
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1-nano
YOUTUBE_API_KEY=...
TRANSCRIPTAPI_API_KEY=
BASE_URL=https://YOUR-SERVICE.up.railway.app
SESSION_SECRET=choose-a-random-long-string
DEMO_AUTH_ENABLED=true
TUBEMIND_DATA_DIR=/app/.local
```

5) Deploy the service
6) Open `/health` to verify the app is up, then open `/`

Notes:
- `DEMO_AUTH_ENABLED=true` skips Google OAuth so the app is easier to grade and demo.
- Mounting the volume at `/app/.local` preserves the SQLite database plus the transcript and LightRAG artifacts across restarts.
- The app now reads Railway's injected `PORT` automatically.
