import os
import io
import threading
import sqlite3
import datetime
import json

from flask import Flask, render_template, request, jsonify, send_file, abort
from model import train_model_background, extract_embedding_for_image, MODEL_PATH

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "attendance.db")
DATASET_DIR = os.path.join(APP_DIR, "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

TRAIN_STATUS_FILE = os.path.join(APP_DIR, "train_status.json")

app = Flask(__name__, static_folder="static", template_folder="templates")

# ---------- DB helpers ----------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    rollno TEXT,
                    class TEXT,
                    section TEXT,
                    phoneno TEXT,
                    created_at TEXT
                )""")
    c.execute("""CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id INTEGER,
                    name TEXT,
                    timestamp TEXT
                )""")
    conn.commit()
    conn.close()

init_db()

# ---------- Train status helpers ----------
def write_train_status(status_dict):
    with open(TRAIN_STATUS_FILE, "w") as f:
        json.dump(status_dict, f)

def read_train_status():
    if not os.path.exists(TRAIN_STATUS_FILE):
        return {"running": False, "progress": 0, "message": "Not trained"}
    with open(TRAIN_STATUS_FILE, "r") as f:
        return json.load(f)

# ensure initial train status file exists
write_train_status({"running": False, "progress": 0, "message": "No training yet."})

# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")

# Dashboard simple API for attendance stats (last 30 days)
@app.route("/attendance_stats")
def attendance_stats():
    import pandas as pd
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT timestamp FROM attendance", conn)
    conn.close()
    if df.empty:
        from datetime import date, timedelta
        days = [(date.today() - datetime.timedelta(days=i)).strftime("%d-%b") for i in range(29, -1, -1)]
        return jsonify({"dates": days, "counts": [0]*30})
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    last_30 = [ (datetime.date.today() - datetime.timedelta(days=i)) for i in range(29, -1, -1) ]
    counts = [ int(df[df['date'] == d].shape[0]) for d in last_30 ]
    dates = [ d.strftime("%d-%b") for d in last_30 ]
    return jsonify({"dates": dates, "counts": counts})

# -------- Add student (form) --------
@app.route("/add_student", methods=["GET", "POST"])
def add_student():
    if request.method == "GET":
        return render_template("add_student.html")
   
    data = request.form
    name = data.get("name","").strip()
    rollno = data.get("rollno","").strip()
    cls = data.get("class","").strip()
    sec = data.get("section","").strip()
    phoneno = data.get("phoneno","").strip()

    errors = {}

    if not name:
        errors["name"] = "Name is required"
    if not rollno:
        errors["rollno"] = "Roll number is required"
    if not cls:
        errors["class"] = "Class is required"
    if not sec:
        errors["section"] = "Section is required"
    if not phoneno:
        errors["phoneno"] = "Phone number is required"

    if errors:
        return jsonify({"errors": errors}), 400

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
#  Check duplicate roll number
    c.execute("SELECT id FROM students WHERE rollno = ?", (rollno,))
    existing = c.fetchone()

    if existing:
        conn.close()
        return jsonify({"errors": {"rollno": "Roll number already exists"}}), 400
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
 
    c.execute("""
        INSERT INTO students (name, rollno, class, section, phoneno, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (name, rollno, cls, sec, phoneno, now))

    sid = c.lastrowid
    conn.commit()
    conn.close()

    return jsonify({"success": True, "student_id": sid})
# -------- Upload face images (after capture) --------
@app.route("/upload_face", methods=["POST"])
def upload_face():
    student_id = request.form.get("student_id")
    if not student_id:
        return jsonify({"error":"student_id required"}), 400
    files = request.files.getlist("images[]")
    saved = 0
    folder = os.path.join(DATASET_DIR, student_id)
    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
    for f in files:
        try:
            fname = f"{datetime.datetime.utcnow().timestamp():.6f}_{saved}.jpg"
            path = os.path.join(folder, fname)
            f.save(path)
            saved += 1
        except Exception as e:
            app.logger.error("save error: %s", e)
    return jsonify({"saved": saved})

# -------- Train model (start background thread) --------
@app.route("/train_model", methods=["GET"])
def train_model_route():
    # if already running, respond accordingly
    status = read_train_status()
    if status.get("running"):
        return jsonify({"status":"already_running"}), 202
    # reset status
    write_train_status({"running": True, "progress": 0, "message": "Starting training"})
    # start background thread
    t = threading.Thread(target=train_model_background, args=(DATASET_DIR, lambda p,m: write_train_status({"running": True, "progress": p, "message": m})))
    t.daemon = True
    t.start()
    return jsonify({"status":"started"}), 202

# -------- Train progress (polling) --------
@app.route("/train_status", methods=["GET"])
def train_status():
    return jsonify(read_train_status())

# -------- Mark attendance page --------
@app.route("/mark_attendance", methods=["GET"])
def mark_attendance_page():
    return render_template("mark_attendance.html")

# -------- Recognize face endpoint (POST image) --------
@app.route("/recognize_face", methods=["POST"])
def recognize_face():
    if "image" not in request.files:
        return jsonify({"recognized": False, "error":"no image"}), 400
    img_file = request.files["image"]
    try:
        emb = extract_embedding_for_image(img_file.stream)
        if emb is None:
            return jsonify({"recognized": False, "error":"no face detected"}), 200
        # attempt prediction
        from model import load_model_if_exists, predict_with_model
        clf = load_model_if_exists()
        if clf is None:
            return jsonify({"recognized": False, "error":"model not trained"}), 200
        pred_label, conf = predict_with_model(clf, emb)
        # threshold confidence
        if conf < 0.5:
            return jsonify({"recognized": False, "confidence": float(conf)}), 200
        # find student name
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT name FROM students WHERE id=?", (int(pred_label),))
        row = c.fetchone()
        name = row[0] if row else "Unknown"

         # save attendance record with timestamp
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO attendance (student_id, name, timestamp) VALUES (?, ?, ?)", 
          (int(pred_label), name, ts))
        conn.commit()
        conn.close()
        return jsonify({"recognized": True, "student_id": int(pred_label), "name": name, "confidence": float(conf)}), 200
    except Exception as e:
        app.logger.exception("recognize error")
        return jsonify({"recognized": False, "error": str(e)}), 500

# -------- Attendance records & filters --------
@app.route("/attendance_record", methods=["GET"])
def attendance_record():
    period = request.args.get("period", "all")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    query = "SELECT id, student_id, name, timestamp FROM attendance"
    params = ()

    today = datetime.date.today()

    if period == "daily":
        query += " WHERE date(timestamp) = ?"
        params = (today.strftime("%Y-%m-%d"),)

    elif period == "weekly":
        start = today - datetime.timedelta(days=today.weekday())
        query += " WHERE date(timestamp) >= ?"
        params = (start.strftime("%Y-%m-%d"),)

    elif period == "monthly":
        start = today.replace(day=1)
        query += " WHERE date(timestamp) >= ?"
        params = (start.strftime("%Y-%m-%d"),)

    query += " ORDER BY timestamp DESC"

    c.execute(query, params)
    records = c.fetchall()

    conn.close()

    return render_template("attendance_record.html", records=records, period=period)
# -------- CSV download --------
@app.route("/download_csv", methods=["GET"])
def download_csv():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, student_id, name, timestamp FROM attendance ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    output = io.StringIO()
    output.write("id,student_id,name,timestamp\n")
    for r in rows:
        output.write(f'{r[0]},{r[1]},{r[2]},{r[3]}\n')
    mem = io.BytesIO()
    mem.write(output.getvalue().encode("utf-8"))
    mem.seek(0)
    return send_file(mem, as_attachment=True, download_name="attendance.csv", mimetype="text/csv")

# -------- Students API for listing/editing --------
@app.route("/students", methods=["GET"])
def students_list():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, rollno, class, section, phoneno, created_at FROM students ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    data = [ {"id":r[0],
              "name":r[1],
              "rollno":r[2],
              "class":r[3],
              "section":r[4],
              "phoneno":r[5],
              "created_at":r[6]
              } 
              for r in rows 
            ]
    return jsonify({"students": data})
#------delete button--------

@app.route("/students/<int:sid>", methods=["DELETE"])
def delete_student(sid):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM students WHERE id=?", (sid,))
    c.execute("DELETE FROM attendance WHERE student_id=?", (sid,))
    conn.commit()
    conn.close()
    # also delete dataset folder
    folder = os.path.join(DATASET_DIR, str(sid))
    if os.path.isdir(folder):
        import shutil
        shutil.rmtree(folder, ignore_errors=True)
    return jsonify({"deleted": True})
#------ students details  ---------
@app.route("/registered_students")
def registered_students_page():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT id, name, rollno, class, section, phoneno, created_at
        FROM students
        ORDER BY id DESC
    """)
    rows = c.fetchall()
    conn.close()

    return render_template("students details.html", students=rows)
#-------- update option ----------
@app.route("/students/<int:sid>", methods=["PUT"])
def update_student(sid):
    data = request.json

    name = data.get("name", "").strip()
    rollno = data.get("rollno", "").strip()
    cls = data.get("class", "").strip()
    sec = data.get("section", "").strip()
    phoneno = data.get("phoneno", "").strip()

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Check duplicate roll number (excluding current student)
    c.execute("SELECT id FROM students WHERE rollno=? AND id!=?", (rollno, sid))
    if c.fetchone():
        conn.close()
        return jsonify({"error": "Roll number already exists"}), 400

    c.execute("""
        UPDATE students
        SET name=?, rollno=?, class=?, section=?, phoneno=?
        WHERE id=?
    """, (name, rollno, cls, sec, phoneno, sid))

    conn.commit()
    conn.close()

    return jsonify({"updated": True})
#------- help ------------
@app.route("/help")
def help_page():
    return render_template("help.html")

# ---------------- run ------------------------
#if __name__ == "__main__":
import webbrowser

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True,use_reloader=False)
    