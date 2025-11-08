# 🚀 Launcher Scripts Guide - Never Use Plain Python Again!

## Quick Reference

| ❌ WRONG | ✅ RIGHT |
|---------|----------|
| `python gradio_app.py` | `.\run_symptom_checker.bat` |
| `python3 gradio_app.py` | `.\run_symptom_checker.ps1` |
| `py gradio_app.py` | `.\symp_env\Scripts\python.exe gradio_app.py` |

---

## 🎯 Option 1: Batch File (EASIEST - Windows)

### File Name
```
run_symptom_checker.bat
```

### How to Use

#### Method A: Double-Click (Fastest!)
1. Open File Explorer
2. Navigate to: `e:\Sympchecker\SympCheck`
3. **Double-click** → `run_symptom_checker.bat`
4. App launches! ✅

#### Method B: Command Line
```powershell
cd e:\Sympchecker\SympCheck
.\run_symptom_checker.bat
```

### What It Does
```
✅ Finds correct Python automatically
✅ Activates virtual environment
✅ Validates setup
✅ Launches app on http://localhost:7860
✅ Shows status messages
```

### Output Example
```
🔍 Checking virtual environment...
✅ Virtual environment found
✅ Starting Symptom Checker...
✅ Access the application at: http://localhost:7860
```

---

## 🎯 Option 2: PowerShell Script

### File Name
```
run_symptom_checker.ps1
```

### How to Use

```powershell
cd e:\Sympchecker\SympCheck
.\run_symptom_checker.ps1
```

### What It Does
```
✅ Colored status messages (better readability)
✅ Comprehensive error reporting
✅ Environment validation
✅ Professional output
✅ Auto-launches browser
```

### Output Example
```
[✓] PowerShell script loaded
[✓] Python environment validated
[✓] Starting Symptom Checker...
[→] Access at: http://localhost:7860
```

### Note
If you get execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 🎯 Option 3: Direct Command (For Developers)

### How to Use
```powershell
cd e:\Sympchecker\SympCheck
.\symp_env\Scripts\python.exe gradio_app.py
```

### When to Use
- Development/testing
- Direct control over execution
- Debugging
- Custom parameters

### Advantages
- ✅ Direct access to output
- ✅ Full control
- ✅ No wrapper scripts
- ✅ Easy to customize

---

## 🎯 Option 4: Activate VEnv First (Standard)

### How to Use
```powershell
cd e:\Sympchecker\SympCheck
.\symp_env\Scripts\Activate.ps1
python gradio_app.py
```

### What Happens
1. Activates virtual environment
2. Updates PowerShell prompt (shows `(symp_env)`)
3. `python` command now uses venv Python
4. App launches

### Output Example
```powershell
(symp_env) PS E:\Sympchecker\SympCheck> python gradio_app.py
Gradio app initialized with lazy loading
Models loaded
[OK] Starting Symptom Checker...
```

### To Deactivate Later
```powershell
deactivate
```

---

## 📊 Comparison Table

| Method | Ease | Speed | Best For |
|--------|------|-------|----------|
| **Batch File** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Daily use |
| **PowerShell Script** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Development |
| **Direct Command** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Testing |
| **Activate VEnv** | ⭐⭐⭐ | ⭐⭐⭐⭐ | Standard workflow |

---

## ⚡ Keyboard Shortcuts

### Stop the App
```
Press: Ctrl + C
```

### Access Web UI
```
Browser URL: http://localhost:7860
```

### Check if Running
```powershell
Get-Process python
```

### Kill All Python Processes
```powershell
Get-Process python | Stop-Process -Force
```

---

## 🔧 Troubleshooting

### Problem: "Module not found: gradio"
```
❌ You used: python gradio_app.py
✅ Use instead: .\run_symptom_checker.bat
```

### Problem: Port 7860 Already in Use
```powershell
# Kill old process
Get-Process python | Stop-Process -Force

# Wait a moment
Start-Sleep -Seconds 2

# Launch app again
.\run_symptom_checker.bat
```

### Problem: PowerShell Execution Policy Error
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
Y  # Type Y and press Enter
```

### Problem: Can't Find Virtual Environment
```powershell
# Check if it exists
Test-Path ".\symp_env\Scripts\python.exe"

# If false, recreate it
python -m venv symp_env
.\symp_env\Scripts\pip install -r requirements.txt
```

---

## 🎯 Recommended Workflow

### Daily Use (Easiest)
```
1. Double-click: run_symptom_checker.bat
2. Wait for: "Running on local URL: http://0.0.0.0:7860"
3. Browser opens automatically
4. Use Symptom Checker! 🎉
```

### Development
```
1. Open PowerShell
2. Run: .\run_symptom_checker.ps1
3. Detailed logs in console
4. Easy to restart
```

### Quick Test
```
1. Run: .\symp_env\Scripts\python.exe gradio_app.py
2. Direct output
3. Full control
```

---

## 📌 Important Notes

### Why NOT Use Plain `python`?
```
System Python (installed globally)
└─ Does NOT have: gradio, torch, faiss, etc.
└─ Result: ModuleNotFoundError

Virtual Environment Python (symp_env)
└─ DOES have: ALL custom packages
└─ Result: App works perfectly! ✅
```

### Key Difference
```
❌ python              → C:\Users\...\Python311\python.exe (missing packages)
✅ .\symp_env\Scripts\python.exe → E:\Sympchecker\SympCheck\symp_env\Scripts\python.exe (has all packages)
```

### Once You Activate VEnv
```powershell
.\symp_env\Scripts\Activate.ps1
# Now you CAN use plain python because it's activated
python gradio_app.py  # ✅ Works! (after activation)
```

---

## 🎁 Bonus: Create Desktop Shortcut

### Method 1: Right-Click File
1. Right-click `run_symptom_checker.bat`
2. Select "Send to" → "Desktop (create shortcut)"
3. Done! Double-click shortcut to launch

### Method 2: Manual Shortcut
1. Right-click Desktop
2. New → Shortcut
3. Location: `e:\Sympchecker\SympCheck\run_symptom_checker.bat`
4. Name: "Symptom Checker"
5. Click Finish

### Result
```
Desktop
├── Symptom Checker (shortcut)
    └─ Double-click to launch app!
```

---

## ✅ Verification Checklist

Before launching, verify:

```powershell
# Check virtual environment exists
Test-Path ".\symp_env\Scripts\python.exe"
# Expected: True

# Check gradio installed
.\symp_env\Scripts\python.exe -c "import gradio; print(f'Gradio {gradio.__version__}')"
# Expected: Gradio 4.19.2

# Check app file exists
Test-Path ".\gradio_app.py"
# Expected: True

# Check launcher scripts exist
Test-Path ".\run_symptom_checker.bat"
Test-Path ".\run_symptom_checker.ps1"
# Expected: Both True
```

---

## 🚀 Summary

| What | Command | Status |
|------|---------|--------|
| **Fastest** | `.\run_symptom_checker.bat` | ✅ Recommended |
| **Best Feedback** | `.\run_symptom_checker.ps1` | ✅ Good |
| **Direct Control** | `.\symp_env\Scripts\python.exe gradio_app.py` | ✅ Advanced |
| **Standard** | Activate venv, then `python` | ✅ Professional |

---

## 📞 Need Help?

### Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| ModuleNotFoundError | Use launcher scripts instead of plain `python` |
| Port already in use | Kill Python: `Get-Process python \| Stop-Process -Force` |
| Execution policy error | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| Can't find python | Check venv exists: `Test-Path ".\symp_env\Scripts\python.exe"` |
| App won't start | Try direct command: `.\symp_env\Scripts\python.exe gradio_app.py` |

---

## 🎉 You're Ready!

**Never use plain `python` again!** 

Use one of these instead:
1. ✅ `.\run_symptom_checker.bat` (easiest)
2. ✅ `.\run_symptom_checker.ps1` (best feedback)
3. ✅ `.\symp_env\Scripts\python.exe gradio_app.py` (full control)
4. ✅ Activate venv, then `python gradio_app.py` (standard)

All will work perfectly and avoid module errors! 🚀
