import os
import cv2
import numpy as np
import torch
import tensorflow as tf
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras import layers
from transformers import AutoTokenizer,AutoModelForCausalLM

CONFIG = {
        "yolo_model_path":      r"yolo26n.pt",
        "unet_model_path":      r"models/custom_scratch_unet.keras",
        "bilstm_model_path":    r"models\bilstm_intent_model",
        "transformer_path":     r"Qwen/Qwen2.5-0.5B-Instruct",   # or "google/flan-t5-base"
        "test_image_path":      r"dataset/train/images/13_013_jpg.rf.1920fe478d505d08d8eb5e96c80c2260.jpg",
        "unet_input_size":      (256, 256),   # (H, W) your UNet was trained on
        "bilstm_max_len":       15,           # max token length your Bi-LSTM uses
        "bilstm_num_classes":   3,            # 0, 1, 2 → update if you have more
          "vectorizer_vocab_path": r"models\vectorizer_vocab.json",  # ← ADD
        "vocab_size":  1000,   # ← must match training
        "max_length":  15,     # ← must match training
    }
     
    # ============================================================
    # 📌 CLASS MAPS — Update labels to match your training data
    # ============================================================
    # ✅ Update this at the top of your pipeline
BILSTM_CLASS_MAP = {
        0: ("Check Safety",     "⚠️"),
        1: ("Summarize Scene",  "🔍"),
        2: ("Query Objects",    "🎯"),
    }



def load_all_models(cfg):
        print("\n📦 Loading all models...\n")
    
        print("  [1/4] Loading YOLO...")
        yolo = YOLO(cfg["yolo_model_path"])
        
    
        print("  [2/4] Loading UNet...")
        unet = load_model(cfg["unet_model_path"], compile=False)
    
        
        print("  [3/4] Loading Bi-LSTM (SavedModel format)...")
        
        # ✅ Correct loader for save_format="tf" (SavedModel folder)
        bilstm = tf.keras.models.load_model(
            cfg["bilstm_model_path"],    # ← folder name, no extension
            compile=False
        )
        _ = bilstm.predict(tf.constant(["warmup"]), verbose=0)
        print("   ✅ Bi-LSTM loaded!")
    
        # Extract vectorizer from inside the model
        # Extract vectorizer from inside the model
        vectorizer = None
        for layer in bilstm.layers:
            if isinstance(layer, layers.TextVectorization):
                vectorizer = layer
                print(f"   ✅ Found TextVectorization: '{layer.name}'")
                print(f"   Vocab size: {len(layer.get_vocabulary())}")
                break
    
        if vectorizer is None:
            print("   ⚠️ Loading vectorizer from vocab JSON...")
            vectorizer = load_vectorizer(
                cfg["vectorizer_vocab_path"],
                vocab_size=cfg["vocab_size"],
                max_length=cfg["max_length"]
            )
       
        
        
    
        # ... inside load_all_models ...
        print("  [4/4] Loading Qwen Lightweight Transformer...")
    
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["transformer_path"],
            trust_remote_code=True
        )
        
        phi3 = AutoModelForCausalLM.from_pretrained(
            cfg["transformer_path"],
            device_map="cuda",          # ✅ Forces strictly onto the GPU
            torch_dtype=torch.float16,         # ✅ FIXED: Was 'dtype', which was being ignored
            attn_implementation="sdpa",        # ✅ Cuts VRAM usage in half
            trust_remote_code=True
        )
        
        phi3.eval()
    
        print("\n✅ All models loaded!\n")
        return yolo, unet, bilstm, vectorizer, tokenizer, phi3
        print("\n✅ All models loaded!\n")
        return yolo, unet, bilstm, vectorizer, tokenizer,phi3



def run_yolo(yolo_model, image_path):
        print("🔍 Running YOLO detection...")
        results = yolo_model(image_path, imgsz=320,verbose=False)
        result   = results[0]
     
        # Annotated image (BGR numpy array)
        yolo_img = result.plot()
     
        # Build a clean text summary of detections
        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id     = int(box.cls[0])
                confidence = float(box.conf[0])
                label      = yolo_model.names[cls_id]
                detections.append(f"{label} ({confidence:.0%})")
        
        detection_str = ", ".join(detections) if detections else "no objects detected"
        print(f"   Detected: {detection_str}")
        return yolo_img, detection_str, detections
     
     
    # ============================================================
    # 3. UNET PREDICTION
    # ============================================================
def run_unet_array(unet_model, orig, input_size):
        """Same as run_unet() below, but operates directly on an in-memory
        BGR frame (np.ndarray) instead of a file path. Use this for video /
        webcam pipelines so every frame avoids a disk write+read round trip."""
        img  = cv2.resize(orig, (input_size[1], input_size[0]))
        inp  = img.astype(np.float32) / 255.0
        inp  = np.expand_dims(inp, axis=0)          # (1, H, W, 3)

        mask_pred = unet_model.predict(inp, verbose=0)[0]  # (H, W, C) or (H, W, 1)

        if mask_pred.shape[-1] == 1:
            mask = (mask_pred[:, :, 0] > 0.5).astype(np.uint8) * 255
        else:
            mask = np.argmax(mask_pred, axis=-1).astype(np.uint8)
            mask = (mask * (255 // max(mask.max(), 1))).astype(np.uint8)

        binary    = (mask > 127).astype(np.uint8)
        drivable  = round(binary.sum() / binary.size * 100, 1)

        mask_rgb  = cv2.cvtColor(
            cv2.resize(mask, (orig.shape[1], orig.shape[0])),
            cv2.COLOR_GRAY2BGR
        )
        overlay   = cv2.addWeighted(orig, 0.6, mask_rgb, 0.4, 0)

        return overlay, mask, drivable


def run_unet(unet_model, image_path, input_size):
        print("🗺️  Running UNet segmentation...")
        orig = cv2.imread(image_path)
        img  = cv2.resize(orig, (input_size[1], input_size[0]))
        inp  = img.astype(np.float32) / 255.0
        inp  = np.expand_dims(inp, axis=0)          # (1, H, W, 3)
     
        mask_pred = unet_model.predict(inp, verbose=0)[0]  # (H, W, C) or (H, W, 1)
     
        # Handle binary or multi-class masks
        if mask_pred.shape[-1] == 1:
            mask = (mask_pred[:, :, 0] > 0.5).astype(np.uint8) * 255
        else:
            mask = np.argmax(mask_pred, axis=-1).astype(np.uint8)
            mask = (mask * (255 // max(mask.max(), 1))).astype(np.uint8)
     
        # Compute drivable area %
        binary    = (mask > 127).astype(np.uint8)
        drivable  = round(binary.sum() / binary.size * 100, 1)
     
        # Overlay mask on original image
        mask_rgb  = cv2.cvtColor(
            cv2.resize(mask, (orig.shape[1], orig.shape[0])),
            cv2.COLOR_GRAY2BGR
        )
        overlay   = cv2.addWeighted(orig, 0.6, mask_rgb, 0.4, 0)
     
        print(f"   Drivable area: {drivable}%")
        return overlay, mask, drivable
     
     


    # ============================================================
    # LOAD VECTORIZER (add this inside load_all_models)
    # ============================================================
def load_vectorizer(vocab_path, vocab_size=1000, max_length=15):
        print("  [3b] Loading TextVectorization vocabulary...")
        import json
        from tensorflow.keras import layers
    
        with open(vocab_path, "r") as f:
            vocab = json.load(f)
    
        vectorize_layer = layers.TextVectorization(
            max_tokens=vocab_size,
            output_mode='int',
            output_sequence_length=max_length
        )
        # Rebuild the exact same layer with saved vocab
        vectorize_layer.set_vocabulary(vocab)
        return vectorize_layer
    
    
    # ============================================================
    # 4. BI-LSTM PREDICTION  (replace the old one)
    # ============================================================
def run_bilstm(bilstm_model, vectorizer, user_query, num_classes):
        print("🧠 Running Bi-LSTM classification...")
    
        # ✅ Exactly like training: tf.constant([string])  shape: (1,)
        inp   = tf.constant([user_query])
        probs = bilstm_model(tf.constant([user_query]), training=False).numpy()[0]                     # shape: (3,)
        class_idx  = int(np.argmax(probs))
        confidence = float(probs[class_idx])
    
        label, emoji = BILSTM_CLASS_MAP.get(class_idx, (f"CLASS_{class_idx}", "❓"))
    
        print(f"   Intent: {emoji} {label} ({confidence:.0%} confidence)")
        return class_idx, label, emoji, confidence, probs



def run_transformer(tokenizer, t5_model, user_query, yolo_detections,
                        drivable_pct, bilstm_label):
        print("🤖 Running Transformer (generating decision)...")
        
        # Build rich telemetry string from all upstream model outputs
       
     
        vision_context = (
            f"Drivable area: {drivable_pct}%. "
            f"YOLO detected: {', '.join(yolo_detections) if yolo_detections else 'nothing'}. "
            
            f"Bi-LSTM intent classification: {bilstm_label}."
        )
        
         
    
        labels = [
            d.split("(")[0].strip().lower()
            for d in yolo_detections
        ]
    
        counts = Counter(labels)
    
        object_text = ", ".join(
            f"{v} {k}{'s' if v > 1 else ''}"
            for k, v in counts.items()
        )
    
        if not object_text:
            object_text = "No objects detected"
    
        messages = [
            {
                "role": "system",
                "content":
                "You are an autonomous driving assistant."
            },
            {
                "role": "user",
                "content": f"""
    Intent:
    {bilstm_label}
    
    Detected Objects:
    {object_text}
    
    Drivable Area:
    {drivable_pct}%
    
    User Question:
    {user_query}
    
    Provide a helpful natural language response.
    """
            }
        ]
    
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        ).to(t5_model.device) # ✅ FIXED
    
        with torch.no_grad():
            outputs = t5_model.generate( 
                **inputs,
                max_new_tokens=20,
                
                do_sample=False,
            
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
    
        response = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
    
       
        if "assistant\n" in response:
            response = response.split("assistant\n")[-1].strip()
        else:
            # Fallback just in case
            response = response.split("User Question:")[-1].split("assistant")[-1].strip()
    
        decision = make_natural_decision(
            response,
            bilstm_label,
            yolo_detections,
            drivable_pct
        )
        return decision,vision_context
    



from collections import Counter
def make_natural_decision(raw, bilstm_label, detections, drivable):
    
        add = ""
    
        if "check safety" in bilstm_label.lower():
    
            objects = [d.split("(")[0].strip() for d in detections] if detections else ["obstacle"]
            obj_str = " and ".join(objects[:2])
    
            add = (
                f"⚠️ STOP IMMEDIATELY. {obj_str} detected in path. "
                f"Drivable area is only {drivable}%. Do not proceed."
            )
    
        elif "summarize" in bilstm_label.lower():
    
            det_str = ", ".join(detections[:3]) if detections else "no objects"
    
            add = (
                f"🔍 Scene summary: {det_str} detected. "
                f"Drivable area is {drivable}%. Proceed with caution."
            )
    
        elif "query objects" in bilstm_label.lower():
    
            det_str = ", ".join(detections) if detections else "no objects detected"
    
            add = (
                f"🎯 Objects in scene: {det_str}. "
                f"Drivable area: {drivable}%."
            )
    
        elif drivable >= 40:
    
            add = (
                f"✅ Path appears clear ({drivable}% drivable). "
                f"Safe to proceed. Monitor surroundings."
            )
    
        else:
    
            add = (
                f"⚠️ Drivable area is {drivable}%. "
                f"Proceed with caution."
            )
    
        # Clean transformer output
        raw = raw.strip()
    
        # Return both
        if raw:
            return f"{add}\n\n🤖 AI Reasoning: {raw}"
    
        return add



def visualize_all(image_path, yolo_img, unet_overlay, unet_mask,
                      user_query, bilstm_label, bilstm_emoji,
                      bilstm_conf, bilstm_probs, yolo_detections,
                      drivable_pct, transformer_decision, vision_context):
     
        orig = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        yolo_rgb  = cv2.cvtColor(yolo_img,    cv2.COLOR_BGR2RGB)
        unet_rgb  = cv2.cvtColor(unet_overlay, cv2.COLOR_BGR2RGB)
     
        fig = plt.figure(figsize=(20, 12), facecolor="#0d1117")
        fig.suptitle("🚗 Autonomous Driving Pipeline — Full Prediction Output",
                     color="white", fontsize=16, fontweight="bold", y=0.98)
     
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.3)
     
        # ── Row 1: Images ─────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(orig)
        ax1.set_title("📷 Original Image", color="white", fontsize=11)
        ax1.axis("off")
     
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(yolo_rgb)
        ax2.set_title("🔍 YOLO Detection", color="#00d4ff", fontsize=11)
        ax2.axis("off")
     
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(unet_rgb)
        ax3.set_title(f"🗺️ UNet Segmentation ({drivable_pct}% drivable)",
                      color="#a0ff80", fontsize=11)
        ax3.axis("off")
     
        # ── Row 2: Bi-LSTM + UNet mask ────────────────────────
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.imshow(unet_mask, cmap="plasma")
        ax4.set_title("🗺️ UNet Raw Mask", color="#a0ff80", fontsize=11)
        ax4.axis("off")
     
        ax5 = fig.add_subplot(gs[1, 1])
        class_names = [BILSTM_CLASS_MAP.get(i, (f"Class {i}", ""))[0]
                       for i in range(len(bilstm_probs))]
        colors = ["#ff4444" if i == np.argmax(bilstm_probs) else "#444466"
                  for i in range(len(bilstm_probs))]
        bars = ax5.barh(class_names, bilstm_probs, color=colors)
        ax5.set_xlim(0, 1)
        ax5.set_facecolor("#0d1117")
        ax5.tick_params(colors="white")
        ax5.set_title(f"🧠 Bi-LSTM: {bilstm_emoji} {bilstm_label} ({bilstm_conf:.0%})",
                      color="#ffaa00", fontsize=11)
        for bar, prob in zip(bars, bilstm_probs):
            ax5.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                     f"{prob:.0%}", va="center", color="white", fontsize=9)
     
        # YOLO detection list
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.set_facecolor("#0d1117")
        ax6.axis("off")
        ax6.set_title("🔍 YOLO Detections", color="#00d4ff", fontsize=11)
        if yolo_detections:
            for i, det in enumerate(yolo_detections[:8]):
                ax6.text(0.05, 0.85 - i*0.12, f"• {det}", color="white",
                         fontsize=10, transform=ax6.transAxes)
        else:
            ax6.text(0.05, 0.5, "No objects detected", color="#888888",
                     fontsize=11, transform=ax6.transAxes)
     
        # ── Row 3: Transformer output (full width) ────────────
        ax7 = fig.add_subplot(gs[2, :])
        ax7.set_facecolor("#1a1f2e")
        ax7.axis("off")
     
        # Color box based on urgency
        if "STOP" in transformer_decision or "⚠️" in transformer_decision:
            box_color = "#3a1a1a"
            title_color = "#ff6666"
        elif "✅" in transformer_decision:
            box_color = "#1a3a1a"
            title_color = "#66ff66"
        else:
            box_color = "#1a2a3a"
            title_color = "#66aaff"
     
        ax7.set_facecolor(box_color)
        ax7.text(0.5, 0.85, f'🗣️ USER QUERY: "{user_query}"',
                 color="#cccccc", fontsize=11, ha="center",
                 transform=ax7.transAxes, style="italic")
        ax7.text(0.5, 0.55, "🤖 TRANSFORMER DECISION:",
                 color=title_color, fontsize=12, fontweight="bold",
                 ha="center", transform=ax7.transAxes)
        ax7.text(0.5, 0.25, transformer_decision,
                 color="white", fontsize=13, fontweight="bold",
                 ha="center", va="center", transform=ax7.transAxes,
                 wrap=True)
        ax7.set_title("🚗 Final Autonomous Driving Decision",
                      color=title_color, fontsize=13, fontweight="bold")
     
        plt.savefig("pipeline_output.png", dpi=150, bbox_inches="tight",
                    facecolor="#0d1117")
        plt.show()
        print("\n💾 Saved: pipeline_output.png")
     


import time
def run_pipeline(user_query: str):
        cfg = CONFIG
    
        # ✅ Unpack 6 values now (added vectorizer)
        yolo, unet, bilstm, vectorizer, tokenizer, phi3 = load_all_models(cfg)
    
        print("=" * 60)
        print(f"📸 Image : {cfg['test_image_path']}")
        print(f"🗣️  Query : {user_query}")
        print("=" * 60)
    
        # Step 1: YOLO
        
    
        start = time.perf_counter()
        yolo_img, detection_str, detections = run_yolo(
            yolo, cfg["test_image_path"]
        )
        end = time.perf_counter()
        
        print(f"Time elapsed: For YOLO  {end - start:.6f} seconds")
     
        # Step 2: UNet
        start = time.perf_counter()
        unet_overlay, unet_mask, drivable_pct = run_unet(
            unet, cfg["test_image_path"], cfg["unet_input_size"]
        )
        end = time.perf_counter()
        
        print(f"Time elapsed: For UNET  {end - start:.6f} seconds")
    
        # Step 3: Bi-LSTM  ✅ pass vectorizer
        start=time.perf_counter()
        class_idx, bilstm_label, bilstm_emoji, bilstm_conf, bilstm_probs = run_bilstm(
            bilstm, vectorizer, user_query, cfg["bilstm_num_classes"]
        )
        end = time.perf_counter()
        
        print(f"Time elapsed: For BIlstm  {end - start:.6f} seconds")
    
        # Step 4: Transformer
        start=time.perf_counter()
        decision, vision_context = run_transformer(
            tokenizer, phi3, user_query, detections, drivable_pct, bilstm_label
        )
        end = time.perf_counter()
        
        print(f"Time elapsed: For transformer  {end - start:.6f} seconds")
    
        # Step 5: Print summary
        print("\n" + "=" * 60)
        print("📊 FULL PIPELINE RESULTS")
        print("=" * 60)
        print(f"🗣️  User Query      : {user_query}")
        print(f"🔍 YOLO Detected   : {detection_str}")
        print(f"🗺️  Drivable Area   : {drivable_pct}%")
        print(f"🧠 Bi-LSTM Class   : {bilstm_emoji} {bilstm_label} ({bilstm_conf:.0%})")
        print(f"📡 Sensor Summary  : {vision_context}")
        print(f"🤖 AI Decision     : {decision}")
        print("=" * 60)
    
        # Step 6: Visualize
        visualize_all(
            cfg["test_image_path"],
            yolo_img, unet_overlay, unet_mask,
            user_query, bilstm_label, bilstm_emoji,
            bilstm_conf, bilstm_probs, detections,
            drivable_pct, decision, vision_context
        )
    
        # Cleanup
        # Cleanup
        del phi3 # ✅ FIXED
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        import gc; gc.collect()