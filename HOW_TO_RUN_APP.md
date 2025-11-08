# 🚀 FIX: How to Run Symptom Checker - PERMANENT SOLUTION

## The Problem

When you run `python gradio_app.py`, you get:
```
ModuleNotFoundError: No module named 'gradio'
```

**Root Cause**: You're using the system Python, not the virtual environment where Gradio is installed.

---

## ✅ The Solution (3 Options)

### **OPTION 1: Use the Batch Script (Easiest for Windows)**

#### Step 1: Run the batch file
Double-click this file in your Explorer:
```
e:\Sympchecker\SympCheck\run_symptom_checker.bat
```

Or from PowerShell/CMD:
```
e:\Sympchecker\SympCheck\run_symptom_checker.bat
```

**What it does:**
- ✅ Automatically finds the correct Python
- ✅ Checks if virtual environment exists
- ✅ Starts the app
- ✅ Shows clear status messages

**Advantage**: Easiest - just double-click and go!

---

### **OPTION 2: Use the PowerShell Script**

#### Step 1: Open PowerShell
```powershell
# Navigate to the project
cd e:\Sympchecker\SympCheck
```

#### Step 2: Run the script
```powershell
.\run_symptom_checker.ps1
```

**What it does:**
- ✅ Shows color-coded messages
- ✅ Validates environment
- ✅ Starts the app
- ✅ Better error messages

**Advantage**: More detailed feedback

---

### **OPTION 3: Use the Full Path (Always Works)**

#### Step 1: Open PowerShell or CMD

#### Step 2: Run with full path
```powershell
E:\Sympchecker\SympCheck\symp_env\Scripts\python.exe gradio_app.py
```

Or shorter version:
```powershell
cd e:\Sympchecker\SympCheck
.\symp_env\Scripts\python.exe gradio_app.py
```

**Advantage**: Direct and reliable

---

### **OPTION 4: Use the venv Activation Script**

#### Step 1: Activate virtual environment
```powershell
cd e:\Sympchecker\SympCheck
.\symp_env\Scripts\Activate.ps1
```

#### Step 2: Run the app
```powershell
python gradio_app.py
```

**Advantage**: After activation, you can use `python` directly

---

## 🎯 Recommended Approach

### **For Regular Use:**
```
💡 Use OPTION 1: run_symptom_checker.bat
   - Just double-click the file
   - App starts automatically
   - No command line needed
```

### **For Development:**
```
💡 Use OPTION 2: run_symptom_checker.ps1
   - More feedback
   - Better error messages
   - Easier troubleshooting
```

### **For Terminal Enthusiasts:**
```
💡 Use OPTION 3: Full path
   - Direct control
   - Perfect for scripting
   - Always works
```

---

## 📋 Quick Reference

### What NOT to do:
```bash
❌ python gradio_app.py            # Wrong Python
❌ py gradio_app.py                # Also wrong
❌ python3 gradio_app.py           # System Python
```

### What to do instead:
```bash
✅ run_symptom_checker.bat         # Recommended (easiest)
✅ .\run_symptom_checker.ps1       # Also good
✅ .\symp_env\Scripts\python.exe gradio_app.py  # Direct path
```

---

## 🔍 Why This Problem Occurred

### Understanding Virtual Environments
```
System Python (❌ No Gradio):
  C:\Users\Admin\AppData\Local\Programs\Python\Python311\python.exe
  └─ Missing all custom packages
  └─ Can't find gradio, torch, etc.

Virtual Environment Python (✅ Has Gradio):
  E:\Sympchecker\SympCheck\symp_env\Scripts\python.exe
  └─ Has Gradio, Torch, FAISS, etc.
  └─ All dependencies installed
```

When you type `python`, Windows uses the system version.
You MUST use the virtual environment version!

---

## ✅ Verification

### To verify virtual environment is set up correctly:

```powershell
# Check if virtual environment exists
Test-Path "e:\Sympchecker\SympCheck\symp_env\Scripts\python.exe"
# Should return: True

# Check if Gradio is installed
E:\Sympchecker\SympCheck\symp_env\Scripts\python.exe -c "import gradio; print('Gradio version:', gradio.__version__)"
# Should show: Gradio version: 4.19.2
```

---

## 🚀 Step-by-Step: First Run

### Using Batch File (Recommended)
1. Open File Explorer
2. Navigate to: `e:\Sympchecker\SympCheck`
3. Double-click: `run_symptom_checker.bat`
4. Watch the messages
5. Wait for: `[OK] Access the application at: http://localhost:7860`
6. Open browser to: `http://localhost:7860`

### Using PowerShell
1. Open PowerShell
2. Run: `cd e:\Sympchecker\SympCheck`
3. Run: `.\run_symptom_checker.ps1`
4. Wait for: `[OK] Starting Symptom Checker on port 7860...`
5. Open browser to: `http://localhost:7860`

### Using Full Path
1. Open PowerShell
2. Run: `E:\Sympchecker\SympCheck\symp_env\Scripts\python.exe gradio_app.py`
3. Wait for: `Running on local URL: http://0.0.0.0:7860`
4. Open browser to: `http://localhost:7860`

---

## 🛠️ Troubleshooting

### Problem: "symp_env not found"
**Solution**: 
- Create virtual environment:
  ```powershell
  cd e:\Sympchecker\SympCheck
  python -m venv symp_env
  .\symp_env\Scripts\pip install -r requirements.txt
  ```

### Problem: "ModuleNotFoundError: No module named 'gradio'"
**Solution**:
- Use correct Python: `.\symp_env\Scripts\python.exe`
- Never use: `python` or `py` or `python3`

### Problem: "Port 7860 already in use"
**Solution**:
- Kill existing Python process:
  ```powershell
  Get-Process python | Stop-Process -Force
  # Wait 2 seconds
  Start-Sleep -Seconds 2
  # Then run the app again
  ```

### Problem: Script won't run (PowerShell)
**Solution**:
- Enable script execution:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

---

## 📚 Understanding the Scripts

### `run_symptom_checker.bat`
```batch
@echo off                          # Hide commands
cd /d "%~dp0"                      # Go to script directory
if not exist "symp_env\..."        # Check if venv exists
"%cd%\symp_env\Scripts\python.exe" # Use venv Python
gradio_app.py                      # Run the app
```

**Good for**: Windows users, quick launches, no terminal knowledge needed

### `run_symptom_checker.ps1`
```powershell
$scriptDir = Split-Path ...        # Get script location
Set-Location $scriptDir             # Change directory
$pythonPath = Join-Path ...         # Build Python path
if (-not (Test-Path $pythonPath))  # Check if exists
& $pythonPath gradio_app.py        # Run with &
```

**Good for**: PowerShell users, better error handling, development

---

## 🔐 Making Permanent

### Add to System PATH (Advanced)
1. Right-click "This PC" → Properties
2. Click "Advanced system settings"
3. Click "Environment Variables"
4. Under "System variables", click "New"
5. Variable name: `PYTHON_SYMPTOM_CHECKER`
6. Variable value: `E:\Sympchecker\SympCheck\symp_env\Scripts\python.exe`
7. Click OK

Then you can use anywhere:
```powershell
%PYTHON_SYMPTOM_CHECKER% gradio_app.py
```

---

## 📝 Command Cheat Sheet

| Task | Command |
|------|---------|
| Start (Batch) | `run_symptom_checker.bat` |
| Start (PowerShell) | `.\run_symptom_checker.ps1` |
| Start (Direct) | `.\symp_env\Scripts\python.exe gradio_app.py` |
| Activate venv | `.\symp_env\Scripts\Activate.ps1` |
| Check Gradio | `.\symp_env\Scripts\python.exe -c "import gradio"` |
| Stop app | Ctrl+C in terminal |
| Kill Python | `Get-Process python \| Stop-Process -Force` |

---

## ✨ Best Practice

### Create a Desktop Shortcut (Windows)

1. Right-click on Desktop
2. New → Shortcut
3. Location: `e:\Sympchecker\SympCheck\run_symptom_checker.bat`
4. Name: "Symptom Checker"
5. Click Finish

Now you can:
- **Double-click the shortcut** to start the app instantly!
- App launches → Browser opens → Ready to use

---

## 🎯 Summary

**Problem**: `ModuleNotFoundError: No module named 'gradio'`

**Root Cause**: Using system Python instead of virtual environment

**Solution**: Use the correct Python:
- ✅ `run_symptom_checker.bat` (Easiest)
- ✅ `.\run_symptom_checker.ps1` (Better feedback)
- ✅ `.\symp_env\Scripts\python.exe gradio_app.py` (Direct)

**Result**: App runs perfectly with all dependencies! 🚀

---

## 📞 Quick Help

| If you see... | Do this... |
|---------------|-----------|
| `ModuleNotFoundError` | Use `run_symptom_checker.bat` |
| `Port already in use` | Kill Python, wait 2s, restart |
| `Script won't run` | Set-ExecutionPolicy RemoteSigned |
| `Can't find gradio_app.py` | Make sure you're in the right folder |

---

**Status**: ✅ **PROBLEM FIXED**

Now you can run Symptom Checker easily without any module errors!
