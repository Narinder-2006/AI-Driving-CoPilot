# 🚗 AI Driving Co-Pilot — QUICK START

## ⚡ 5-Minute Setup

### 1️⃣ Install Dependencies
```bash
cd AI_Car_Assistant
pip install -r requirements.txt
```

### 2️⃣ Run Backend
**Windows:**
```bash
python main.py
```

**Linux/Mac:**
```bash
python3 main.py
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
✅ All models loaded successfully!
```

### 3️⃣ Open Frontend
- Open browser: `http://localhost:8000`
- You should see the AI Co-Pilot interface

### 4️⃣ Start Using!

**Webcam:**
1. Select "Webcam (Live)" in sidebar
2. Click "▶️ Start Webcam"
3. Type a question like "Is it safe?"
4. Click "Send"

**Upload Video:**
1. Select "Upload Video" in sidebar
2. Upload a driving video (MP4, AVI, etc.)
3. Wait for processing
4. Ask questions about any frame

**Test Image:**
1. Select "Test Image"
2. System loads default test image
3. Try questions immediately

---

## 🎮 Example Queries

- ✅ "Is it safe to proceed?"
- 🔍 "What do you see around me?"
- 🎯 "List all detected objects"
- ⚠️ "Are there any obstacles?"
- 📊 "Summarize the driving scene"

---

## ❌ Issues?

### Backend won't start
```
❌ Address already in use
```
**Fix**: Close other processes on port 8000 or edit `main.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=8001)  # Use 8001
```

### Webcam not working
```
❌ Cannot open webcam
```
**Fix**: 
- Check camera permissions
- Try: `python -c "import cv2; print('OK' if cv2.VideoCapture(0).isOpened() else 'FAIL')"`

### Models loading error
```
❌ FileNotFoundError
```
**Fix**: Ensure these files exist in project root:
- ✅ `yolo26n.pt`
- ✅ `custom_scratch_unet.keras`
- ✅ `bilstm_intent_model/` (folder)
- ✅ `vectorizer_vocab.json`
- ✅ `dataset/train/images/` (has test image)

### Video lag/slow
**Solutions** (fastest first):
1. ↓ Lower YOLO size to 256
2. Reduce refresh rate in `script.js` (100 → 200ms)
3. Use smaller video resolution
4. Install CUDA for GPU support

---

## 🚀 Performance Tips

| Setting | Speed | Accuracy |
|---------|-------|----------|
| YOLO 256 | ⚡⚡⚡ Fast | Good |
| YOLO 320 | ⚡⚡ Medium | Very Good |
| YOLO 640 | ⚡ Slow | Excellent |

**Default: 320** (recommended for real-time)

---

## 📁 Important Files

```
AI_Car_Assistant/
├── main.py                    # ← Backend (run this)
├── frontend/
│   ├── index.html            # ← Web interface
│   ├── style.css             # ← Styling
│   ├── script.js             # ← Frontend logic
├── pipeline.py               # ← Model pipeline
├── requirements.txt          # ← Dependencies
```

---

## 🌐 Accessing Remotely

To access from another computer:

1. **Find your IP:**
   ```bash
   # Windows
   ipconfig | findstr "IPv4"
   
   # Linux/Mac
   ifconfig | grep "inet"
   ```

2. **Edit `main.py`:**
   ```python
   # Replace "0.0.0.0" with your actual IP
   uvicorn.run(app, host="192.168.x.x", port=8000)
   ```

3. **Access from other computer:**
   ```
   http://192.168.x.x:8000
   ```

---

## 🎨 Customization

### Change Port
In `main.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=9000)  # ← Change to 9000
```

### Adjust Frame Rate
In `frontend/script.js`:
```javascript
// Slower = less CPU usage
webcamFrameInterval = setInterval(fetchWebcamFrame, 200);  // 5 FPS instead of 10
```

### Change UI Theme
Edit `frontend/style.css` - search for color codes:
- `#00d4ff` — Cyan
- `#a0ff80` — Green
- `#ff4b4b` — Red

---

## 📖 Full Documentation

See **README.md** for complete guide with:
- API endpoints
- Architecture
- Troubleshooting
- Deployment options
- Model customization

---

**Everything ready! Open http://localhost:8000 and start driving! 🚗✨**
