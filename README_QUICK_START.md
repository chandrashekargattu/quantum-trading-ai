# 🚀 Quantum Trading AI - Quick Start

## One-Command Start

Just run this single command to start everything:

```bash
./START_APP.sh
```

That's it! The script will:
- ✅ Automatically use the correct Node.js version
- ✅ Kill any existing processes
- ✅ Install missing dependencies
- ✅ Start both frontend and backend
- ✅ Show you the status
- ✅ Display logs

## Access the Application

Once started, open your browser and go to:
- **Frontend**: http://localhost:3000
- **Backend API Docs**: http://localhost:8000/docs

## Stop the Application

```bash
./STOP_APP.sh
```

## Troubleshooting

If you have any issues:

1. **Node.js version error**: The script automatically handles this
2. **Port already in use**: The script automatically kills existing processes
3. **Dependencies missing**: The script automatically installs them

## Manual Commands (if needed)

```bash
# View logs
tail -f frontend.log
tail -f backend.log

# Check what's running
lsof -i :3000  # Frontend
lsof -i :8000  # Backend

# Manual cleanup
pkill -f "npm run dev"
pkill -f "uvicorn"
```

## First Time Setup

If this is your first time:

1. Make sure you have `nvm` installed
2. Run `./START_APP.sh`
3. Everything else is automatic!

---

**No more complicated steps. Just run `./START_APP.sh` and start trading!** 🎉

