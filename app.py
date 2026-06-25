from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
import uuid
import logging
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from agent_orchestrator import run_agent_from_api
from ppt_writer import wants_acknowledgement
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
 
app = Flask(__name__)

flask_key = os.getenv("FLASK_SECRET_KEY")
if not flask_key:
    logger.warning("FLASK_SECRET_KEY environment variable is missing. Using a fallback key (unsafe for production).")
    app.secret_key = "fallback_secret_key"
else:
    app.secret_key = flask_key
 
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
 
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
 
app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"]      = OUTPUT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB
 
ALLOWED_EXTENSIONS = {"txt", "pdf", "csv", "xlsx", "xls", "docx", "doc", "md"}
 
 
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
 
 
@app.route("/")
def home():
    return render_template("layout.html", chat_history=[], generated_content=None, download_url=None, warnings=[])
 
 
@app.route("/generate", methods=["POST"])
def generate():
    prompt   = request.form.get("prompt", "").strip()
    doc_type = request.form.get("doc_type", "docx")
 
    if not prompt:
        flash("Prompt cannot be empty.", "danger")
        return redirect(url_for("home"))
 
    logger.info(f"FILES IN REQUEST: {list(request.files.keys())}")
    logger.info(f"FILES GETLIST: {[f.filename for f in request.files.getlist('document')]}")
 
    # ── Save uploaded files ───────────────────────────────────────────────────
    saved_paths = []
    rejected    = []
 
    for file in request.files.getlist("document"):
        if not file or not file.filename:
            logger.info("Skipping empty file entry")
            continue
        if not allowed_file(file.filename):
            rejected.append(file.filename)
            continue
        filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        path     = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)
        saved_paths.append(path)
        logger.info(f"Saved: {path} | size={os.path.getsize(path)}")
 
    if rejected:
        flash(
            f"Skipped unsupported file(s): {', '.join(rejected)}. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            "warning"
        )
 
    has_files = len(saved_paths) > 0
    logger.info(f"Total saved: {len(saved_paths)} | has_files={has_files}")
 
    try:
        warnings = []
        result = run_agent_from_api(
            prompt=prompt,
            file_paths=saved_paths,
            output_format=doc_type,
            has_files=has_files,
            add_acknowledgement=wants_acknowledgement(prompt) if doc_type == "pptx" else False,
            warnings_out=warnings
        )
        for w in warnings:
            flash(w, "warning")
 
        # Unpack FIRST — orchestrator returns (path, slide_count) for pptx,
        # plain path for docx/pdf. os.path.basename must never receive a tuple.
        slide_count = None
        if isinstance(result, tuple):
            output_path, slide_count = result
        else:
            output_path = result
 
        output_filename = os.path.basename(output_path)
        n = len(saved_paths)
 
        if slide_count is not None:
            msg = (
                f"✅ Presentation generated from <b>{n} file(s)</b>: "
                f"<b>{output_filename}</b> ({slide_count} slides)"
                if n > 0 else
                f"✅ Presentation generated: <b>{output_filename}</b> ({slide_count} slides)"
            )
        else:
            msg = (
                f"✅ Report generated from <b>{n} file(s)</b>: <b>{output_filename}</b>"
                if n > 0 else
                f"✅ Document generated: <b>{output_filename}</b>"
            )
 
        logger.info(f"Output: {output_filename}")
 
        return render_template(
            "layout.html",
            chat_history=[],
            generated_content=msg,
            download_url=url_for("download", filename=output_filename),
            warnings=warnings
        )
 
    except Exception as e:
        logger.exception("Generation failed")
        err_msg = str(e)
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            err_msg = err_msg.replace(groq_key, "[REDACTED_API_KEY]")
        return render_template(
            "layout.html",
            chat_history=[],
            generated_content=f"❌ Error: {err_msg}",
            download_url=None,
            warnings=[]
        )
    finally:
        for path in saved_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(f"Cleaned up uploaded file: {path}")
            except Exception as cleanup_err:
                logger.warning(f"Failed to delete uploaded file {path}: {cleanup_err}")
 
 
@app.route("/download/<path:filename>")
def download(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)
 
 
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    debug = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1")
    app.run(debug=debug, port=port)
 