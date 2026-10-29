import os
import cv2
import time
import numpy as np
import pandas as pd
from datetime import datetime
import streamlit as st
from PIL import Image

import database as db
from detector import PlateDetectorOCR, DEFAULT_MODEL_PATH, list_available_models

# ----------------- PAGE CONFIG & STYLING -----------------
st.set_page_config(
    page_title="Smart Parking Entry & Exit Logger",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern card styling and high-contrast badges
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        padding: 18px 22px;
        border: 1px solid #334155;
        color: #f8fafc;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .metric-val {
        font-size: 1.85rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .status-badge-active {
        background-color: #059669;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge-exited {
        background-color: #475569;
        color: #e2e8f0;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- INITIALIZATION -----------------
CROPS_DIR = os.path.join(os.path.dirname(__file__), "crops")
os.makedirs(CROPS_DIR, exist_ok=True)
db.init_db()

@st.cache_resource(show_spinner="Loading YOLOv8 plate detector & EasyOCR models...")
def get_detector(weights_path: str):
    return PlateDetectorOCR(model_path=weights_path)

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.title("⚙️ System Control")
    st.markdown("Configure gate mode, parking tariffs, and capacity.")

    gate_mode = st.radio(
        "Select Active Gate Mode:",
        ["🟢 Entry Gate", "🔴 Exit Gate"],
        index=0,
        help="Entry logs vehicles in. Exit calculates duration and fee."
    )
    is_entry_mode = "Entry" in gate_mode

    st.divider()
    st.subheader("Parking Rates & Capacity (₹ INR)")
    total_slots = st.number_input("Total Parking Slots", min_value=1, max_value=1000, value=50, step=5)
    hourly_rate = st.number_input("Hourly Parking Rate (₹)", min_value=5.0, max_value=500.0, value=40.0, step=5.0,
                                  help="Example: ₹40.00/hr (₹20.00 per 30 minutes)")
    min_fee = st.number_input("Base Minimum Fee (₹)", min_value=0.0, max_value=200.0, value=20.0, step=5.0)

    st.divider()
    st.subheader("Model Configuration")
    
    available_models = list_available_models()
    if not available_models:
        available_models = [DEFAULT_MODEL_PATH]

    model_options = {os.path.basename(p): p for p in available_models}
    model_options["Custom Path..."] = "custom"

    selected_model_name = st.selectbox(
        "Trained Plate Detector Model",
        options=list(model_options.keys()),
        index=0,
        help="Select trained weights imported from TrackX or trained locally."
    )

    if model_options[selected_model_name] == "custom":
        custom_model_path = st.text_input("Custom Weights Path", value=DEFAULT_MODEL_PATH)
    else:
        custom_model_path = model_options[selected_model_name]
        st.caption(f"Weights: `{os.path.basename(custom_model_path)}`")

    conf_thresh = st.slider("Detector Confidence Threshold", min_value=0.1, max_value=0.9, value=0.25, step=0.05)

# Load Detector
try:
    detector = get_detector(custom_model_path)
except Exception as e:
    st.error(f"Error loading model from `{custom_model_path}`: {e}")
    st.stop()

# ----------------- HEADER & LIVE SUMMARY -----------------
st.title("🚗 Smart Parking Entry & Exit Logger")
st.markdown("Automated License Plate Recognition (ALPR) system with real-time entry/exit tracking, duration billing, and parking analytics.")

lot_summary = db.get_lot_summary(total_capacity=int(total_slots))

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Capacity</div>
            <div class="metric-val">{lot_summary['total_capacity']}</div>
        </div>
    """, unsafe_allow_html=True)
with col_m2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Occupied Slots</div>
            <div class="metric-val" style="color: #f59e0b;">{lot_summary['occupied']}</div>
        </div>
    """, unsafe_allow_html=True)
with col_m3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Available Slots</div>
            <div class="metric-val" style="color: #10b981;">{lot_summary['available']}</div>
        </div>
    """, unsafe_allow_html=True)
with col_m4:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Today's Revenue</div>
            <div class="metric-val" style="color: #60a5fa;">₹{lot_summary['today_revenue']:.2f}</div>
        </div>
    """, unsafe_allow_html=True)

st.write("")

# ----------------- MAIN INTERFACE -----------------
tab_scan, tab_database, tab_train = st.tabs(["📷 Gate Recognition & Action", "📊 Parking Activity Records", "🧠 Model Training & Data"])

with tab_scan:
    st.subheader(f"{'🟢 Vehicle Entry Processing' if is_entry_mode else '🔴 Vehicle Exit & Checkout'}")

    input_source = st.radio(
        "Choose Camera / Input Stream:",
        ["Static Image Upload", "Live Webcam Snapshot", "🎥 Auto-Scan Video Stream"],
        horizontal=True
    )

    frame_to_process = None

    if input_source == "Static Image Upload":
        uploaded_file = st.file_uploader("Upload Vehicle Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame_to_process = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    elif input_source == "Live Webcam Snapshot":
        cam_input = st.camera_input("Capture Plate Snapshot from Webcam")
        if cam_input is not None:
            image_bytes = np.asarray(bytearray(cam_input.read()), dtype=np.uint8)
            frame_to_process = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    elif input_source == "🎥 Auto-Scan Video Stream":
        video_file = st.file_uploader("Upload Video File for Automatic Plate Detection (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])
        if video_file is not None:
            tfile_path = os.path.join(CROPS_DIR, "temp_video.mp4")
            with open(tfile_path, "wb") as f:
                f.write(video_file.read())

            st.info("Video uploaded! Click **Start Automatic Video Detection** to stream and automatically extract plates.")

            col_vcfg1, col_vcfg2 = st.columns(2)
            with col_vcfg1:
                frame_skip = st.slider("Process Every Nth Frame (Speed vs Accuracy)", min_value=1, max_value=20, value=5, help="Higher values process faster by skipping redundant video frames.")
            with col_vcfg2:
                auto_log = st.checkbox("Automatically log plates into database upon detection", value=True)

            if st.button("▶️ Start Automatic Video Detection", type="primary"):
                vcap = cv2.VideoCapture(tfile_path)
                total_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = vcap.get(cv2.CAP_PROP_FPS) or 25.0

                progress_bar = st.progress(0.0)
                video_placeholder = st.empty()
                status_placeholder = st.empty()

                detected_plates_seen = set()
                frame_idx = 0

                while vcap.isOpened():
                    ret, frame = vcap.read()
                    if not ret:
                        break

                    frame_idx += 1
                    if frame_idx % frame_skip != 0:
                        continue

                    # Run detection & OCR
                    detections = detector.detect_and_recognize(frame, conf_thresh=conf_thresh)
                    annotated_frame = detector.draw_detections(frame, detections)

                    # Update video preview in real time
                    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(rgb_frame, channels="RGB", width="stretch")
                    progress_bar.progress(min(1.0, frame_idx / max(1, total_frames)))

                    # Process any detected plates
                    for det in detections:
                        plate_text = det["plate_text"]
                        if plate_text and len(plate_text) >= 4 and plate_text not in detected_plates_seen:
                            detected_plates_seen.add(plate_text)

                            # Save crop
                            crop_fname = f"plate_{int(time.time())}_{plate_text}.jpg"
                            crop_fpath = os.path.join(CROPS_DIR, crop_fname)
                            cv2.imwrite(crop_fpath, det["plate_crop"])

                            if auto_log:
                                if is_entry_mode:
                                    success, msg, _ = db.log_entry(plate_text, crop_fpath)
                                else:
                                    success, msg, _ = db.log_exit(plate_text, rate_per_hour=hourly_rate, min_fee=min_fee)
                                
                                if success:
                                    status_placeholder.success(f"🎯 Auto-Detected & Logged: **{plate_text}** ({msg})")
                                else:
                                    status_placeholder.warning(f"ℹ️ Auto-Detected: **{plate_text}** ({msg})")
                            else:
                                status_placeholder.info(f"🎯 Detected Plate: **{plate_text}** (Auto-log disabled)")

                vcap.release()
                st.success(f"🏁 Finished scanning video! Detected {len(detected_plates_seen)} unique plates: {', '.join(detected_plates_seen) if detected_plates_seen else 'None'}")
                time.sleep(2)
                st.rerun()

    # Processing block
    if frame_to_process is not None:
        col_view, col_action = st.columns([1.6, 1.0])

        with st.spinner("Detecting license plate & extracting characters..."):
            detections = detector.detect_and_recognize(frame_to_process, conf_thresh=conf_thresh)
            annotated_frame = detector.draw_detections(frame_to_process, detections)

        with col_view:
            st.markdown("#### Annotated Gate View")
            # Convert BGR to RGB for Streamlit display
            rgb_annotated = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st.image(rgb_annotated, width="stretch")

        with col_action:
            st.markdown("#### Plate OCR & Gate Action")
            
            if len(detections) == 0:
                st.warning("⚠️ No license plate detected in this frame.")
                manual_input = st.text_input("Enter Plate Number Manually (Optional):").strip().upper()
                selected_plate = manual_input if manual_input else None
                crop_path_to_save = None
            else:
                det = detections[0]
                detected_text = det["plate_text"]

                # Display crops
                sub_c1, sub_c2 = st.columns(2)
                with sub_c1:
                    rgb_crop = cv2.cvtColor(det["plate_crop"], cv2.COLOR_BGR2RGB)
                    st.image(rgb_crop, caption="Cropped Plate", width="stretch")
                with sub_c2:
                    st.image(det["preprocessed_crop"], caption="Preprocessed OCR", width="stretch")

                st.write(f"**Detector Conf:** `{det['detector_conf']:.2f}` | **OCR Conf:** `{det['ocr_confidence']:.2f}`")
                
                # Allow user to verify or adjust OCR recognized string
                verified_plate = st.text_input(
                    "Recognized Plate String (Edit if needed):",
                    value=detected_text,
                    help="Clean alphanumeric string recognized by EasyOCR"
                ).strip().upper()

                selected_plate = verified_plate if verified_plate else None

                # Save crop image
                crop_filename = f"plate_{int(time.time())}.jpg"
                crop_path_to_save = os.path.join(CROPS_DIR, crop_filename)
                cv2.imwrite(crop_path_to_save, det["plate_crop"])

            # Action Button
            st.write("")
            if is_entry_mode:
                btn_label = "🟢 Confirm Vehicle Entry"
                action_btn = st.button(btn_label, type="primary", width="stretch", disabled=(not selected_plate))
                
                if action_btn and selected_plate:
                    # Check capacity
                    if lot_summary['available'] <= 0:
                        st.error("❌ Parking Lot Full! Cannot log vehicle entry.")
                    else:
                        success, message, record = db.log_entry(selected_plate, crop_path_to_save)
                        if success:
                            st.success(f"✅ {message}")
                            st.balloons()
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.warning(f"⚠️ {message}")

            else:
                btn_label = "🔴 Confirm Vehicle Exit & Checkout"
                action_btn = st.button(btn_label, type="primary", width="stretch", disabled=(not selected_plate))

                if action_btn and selected_plate:
                    success, message, record = db.log_exit(selected_plate, rate_per_hour=hourly_rate, min_fee=min_fee)
                    if success:
                        st.success(f"✅ {message}")
                        st.markdown(f"""
                        <div style="background-color: #1e293b; padding: 15px; border-radius: 8px; border-left: 5px solid #10b981; margin-top: 10px;">
                            <h4 style="margin: 0; color: #10b981;">Receipt Summary</h4>
                            <p style="margin: 4px 0 0 0; color: #e2e8f0;"><b>Plate:</b> {record['plate_number']}</p>
                            <p style="margin: 4px 0 0 0; color: #e2e8f0;"><b>Entry:</b> {record['entry_time']}</p>
                            <p style="margin: 4px 0 0 0; color: #e2e8f0;"><b>Exit:</b> {record['exit_time']}</p>
                            <p style="margin: 4px 0 0 0; color: #e2e8f0;"><b>Duration:</b> {record['duration_minutes']:.1f} mins</p>
                            <h3 style="margin: 8px 0 0 0; color: #38bdf8;">Amount Due: ₹{record['fee_charged']:.2f}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        time.sleep(1.5)
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")

with tab_database:
    st.subheader("📋 Parking Lot History & Live Logs")

    col_filter, col_refresh = st.columns([3, 1])
    with col_filter:
        status_filter = st.selectbox("Filter by Status:", ["ALL", "ACTIVE (Currently Parked)", "EXITED (Completed)"])
    with col_refresh:
        st.write("")
        if st.button("🔄 Refresh Data", width="stretch"):
            st.rerun()

    filter_arg = None
    if "ACTIVE" in status_filter:
        filter_arg = "ACTIVE"
    elif "EXITED" in status_filter:
        filter_arg = "EXITED"

    records = db.get_all_records(status_filter=filter_arg)

    if records:
        df = pd.DataFrame(records)

        # Quick Search Bar
        search_plate = st.text_input("🔍 Search by Plate Number:").strip().upper()
        if search_plate:
            df = df[df["plate_number"].str.contains(search_plate, na=False)]

        # Display Dataframe
        st.dataframe(
            df[["id", "plate_number", "entry_time", "exit_time", "duration_minutes", "fee_charged", "status"]],
            width="stretch",
            height=380
        )

        # CSV Download Button
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Parking Records to CSV",
            data=csv_data,
            file_name=f"parking_records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No parking logs found yet. Start scanning vehicles at Entry Gate!")

with tab_train:
    st.subheader("🧠 Model Training, Weights & Dataset Management")
    st.markdown(
        "Trained models and license plate detection datasets imported from "
        "[TrackX (Satyam3se/TrackX_)](https://github.com/Satyam3se/TrackX_). "
        "Manage weights, inspect dataset samples, or train/fine-tune custom models directly."
    )

    data_dir = os.path.join(os.path.dirname(__file__), "data", "license_plate_dataset")
    train_imgs_dir = os.path.join(data_dir, "train", "images")
    val_imgs_dir = os.path.join(data_dir, "val", "images")
    test_imgs_dir = os.path.join(data_dir, "test", "images")

    num_train = len(os.listdir(train_imgs_dir)) if os.path.exists(train_imgs_dir) else 0
    num_val = len(os.listdir(val_imgs_dir)) if os.path.exists(val_imgs_dir) else 0
    num_test = len(os.listdir(test_imgs_dir)) if os.path.exists(test_imgs_dir) else 0
    models_list = list_available_models()

    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        st.metric("Training Images", f"{num_train:,}")
    with col_t2:
        st.metric("Validation Images", f"{num_val:,}")
    with col_t3:
        st.metric("Test Images", f"{num_test:,}")
    with col_t4:
        st.metric("Available Models", len(models_list))

    st.write("")
    
    col_w_left, col_w_right = st.columns([1, 1])

    with col_w_left:
        st.markdown("### 📦 Available Trained Weights")
        weight_rows = []
        for mpath in models_list:
            fname = os.path.basename(mpath)
            fsize = round(os.path.getsize(mpath) / (1024 * 1024), 2)
            is_active = (mpath == custom_model_path)
            weight_rows.append({
                "Model File": fname,
                "Size (MB)": f"{fsize} MB",
                "Status": "ACTIVE ✅" if is_active else "Ready",
                "Path": mpath
            })
        if weight_rows:
            st.dataframe(pd.DataFrame(weight_rows), width="stretch", hide_index=True)
        else:
            st.warning("No model weights found in models/ folder.")

    with col_w_right:
        st.markdown("### 🎯 Model Fine-Tuning / Training Studio")
        with st.form("train_form"):
            train_base = st.selectbox(
                "Base Weights to Train From:",
                ["yolov8n.pt (Start from Base Nano)"] + [os.path.basename(p) for p in models_list],
                index=0
            )
            base_w_path = "yolov8n.pt" if "yolov8n.pt" in train_base else os.path.join(os.path.dirname(__file__), "models", train_base)
            
            c_ep, c_bs = st.columns(2)
            with c_ep:
                epochs_input = st.number_input("Training Epochs", min_value=1, max_value=100, value=10, step=1)
            with c_bs:
                batch_input = st.number_input("Batch Size", min_value=1, max_value=64, value=8, step=2)

            quick_sanity = st.checkbox("Quick 1-Epoch Sanity Run (fast verification)", value=False)
            
            submitted = st.form_submit_button("🚀 Start Training Pipeline", width="stretch")

        if submitted:
            import train
            status_box = st.empty()
            status_box.info("⏳ Initializing YOLOv8 training pipeline... Please wait.")
            try:
                out_path = train.train_model(
                    data_yaml=os.path.join(os.path.dirname(__file__), "data", "data.yaml"),
                    base_weights=base_w_path,
                    epochs=int(epochs_input),
                    batch=int(batch_input),
                    quick_test=quick_sanity
                )
                if out_path:
                    st.success(f"🎉 Training complete! Saved to `{out_path}`. Selected model updated.")
                    st.cache_resource.clear()
                    st.rerun()
            except Exception as ex:
                st.error(f"❌ Training failed: {ex}")

    st.divider()
    st.markdown("### 🖼️ Dataset Sample & Detection Inspector")
    st.markdown("Inspect sample license plate images from the dataset and verify detection with the active model.")
    
    if os.path.exists(test_imgs_dir):
        test_images = [f for f in os.listdir(test_imgs_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if test_images:
            selected_sample = st.selectbox("Choose Sample Test Image from Dataset:", test_images[:20])
            sample_path = os.path.join(test_imgs_dir, selected_sample)
            
            c_img1, c_img2 = st.columns(2)
            sample_bgr = cv2.imread(sample_path)
            if sample_bgr is not None:
                with c_img1:
                    st.image(cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2RGB), caption="Raw Sample Image", width="stretch")
                
                with c_img2:
                    sample_dets = detector.detect_and_recognize(sample_bgr, conf_thresh=conf_thresh)
                    annotated_sample = detector.draw_detections(sample_bgr, sample_dets)
                    st.image(cv2.cvtColor(annotated_sample, cv2.COLOR_BGR2RGB), caption=f"Detected {len(sample_dets)} License Plate(s)", width="stretch")
                    if sample_dets:
                        for idx, d in enumerate(sample_dets, 1):
                            st.write(f"**Plate {idx}:** `{d['plate_text'] or 'UNRECOGNIZED'}` | **Detector Conf:** `{d['detector_conf']:.2f}` | **OCR Conf:** `{d['ocr_confidence']:.2f}`")
                    else:
                        st.caption("No plates detected with current threshold.")

