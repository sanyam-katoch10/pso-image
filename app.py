import os
import uuid
import json
import queue
import threading
import time
from io import BytesIO
from flask import (
    Flask, request, render_template, send_file,
    jsonify, Response, redirect, url_for
)
import cv2
from pso_engine import segment_image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}
UPLOAD_FOLDER = os.path.join("static", "uploads")
RESULT_FOLDER = os.path.join("static", "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

progress_queues = {}
job_files = {}
CLEANUP_DELAY = 120


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def cleanup_job_files(job_id):
    time.sleep(CLEANUP_DELAY)
    paths = job_files.pop(job_id, [])
    for path in paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


def schedule_cleanup(job_id):
    t = threading.Thread(target=cleanup_job_files, args=(job_id,), daemon=True)
    t.start()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/segment", methods=["POST"])
def segment():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    try:
        num_centroids = int(request.form.get("num_centroids", 3))
        num_centroids = max(2, min(num_centroids, 10))
    except (ValueError, TypeError):
        num_centroids = 3

    color_mode = request.form.get("color", "color")
    color = color_mode == "color"

    job_id = str(uuid.uuid4())[:8]
    ext = file.filename.rsplit(".", 1)[1].lower()
    upload_filename = f"{job_id}_original.{ext}"
    upload_path = os.path.join(UPLOAD_FOLDER, upload_filename)
    file.save(upload_path)

    result_filename = f"{job_id}_segmented.jpg"
    result_path = os.path.join(RESULT_FOLDER, result_filename)

    job_files[job_id] = [upload_path, result_path]
    progress_queues[job_id] = queue.Queue()

    def run_segmentation():
        try:
            def progress_cb(iteration, max_iter, cost, progress):
                try:
                    progress_queues[job_id].put({
                        "progress": round(progress, 1),
                        "iteration": iteration,
                        "cost": round(float(cost), 4) if cost else 0,
                        "status": "processing"
                    })
                except Exception:
                    pass

            result = segment_image(
                upload_path, num_centroids,
                color=color, max_iter=100,
                progress_callback=progress_cb
            )

            cv2.imwrite(result_path, result["segmented_image"])

            progress_queues[job_id].put({
                "progress": 100,
                "status": "complete",
                "result": {
                    "original_image": f"/{upload_path}",
                    "segmented_image": f"/{result_path}",
                    "processing_time": result["processing_time"],
                    "dimensions": result["dimensions"],
                    "num_centroids": result["num_centroids"],
                    "job_id": job_id
                }
            })

            schedule_cleanup(job_id)

        except Exception as e:
            progress_queues[job_id].put({
                "progress": 0,
                "status": "error",
                "error": str(e)
            })
            schedule_cleanup(job_id)

    thread = threading.Thread(target=run_segmentation, daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/progress/<job_id>")
def progress(job_id):
    def generate():
        q = progress_queues.get(job_id)
        if not q:
            yield f"data: {json.dumps({'error': 'Invalid job ID'})}\n\n"
            return

        while True:
            try:
                data = q.get(timeout=30)
                yield f"data: {json.dumps(data)}\n\n"
                if data.get("status") in ("complete", "error"):
                    progress_queues.pop(job_id, None)
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'status': 'waiting'})}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/result/<job_id>")
def result(job_id):
    upload_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.startswith(job_id)]
    result_files = [f for f in os.listdir(RESULT_FOLDER) if f.startswith(job_id)]

    if not upload_files or not result_files:
        return redirect(url_for("index"))

    original_path = f"/static/uploads/{upload_files[0]}"
    segmented_path = f"/static/results/{result_files[0]}"

    return render_template("result.html",
                           original_image=original_path,
                           segmented_image=segmented_path,
                           job_id=job_id)


@app.route("/download/<job_id>")
def download(job_id):
    result_files = [f for f in os.listdir(RESULT_FOLDER) if f.startswith(job_id)]
    if not result_files:
        return jsonify({"error": "File not found"}), 404

    filepath = os.path.join(RESULT_FOLDER, result_files[0])

    with open(filepath, "rb") as f:
        file_data = f.read()

    paths = job_files.pop(job_id, [])
    for path in paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass

    return send_file(BytesIO(file_data), as_attachment=True,
                     download_name=f"segmented_{job_id}.jpg",
                     mimetype="image/jpeg")


if __name__ == "__main__":
    for folder in [UPLOAD_FOLDER, RESULT_FOLDER]:
        for f in os.listdir(folder):
            try:
                os.remove(os.path.join(folder, f))
            except OSError:
                pass

    server_port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=server_port, debug=False)
