import os
from flask import Flask, send_from_directory
from routers.app_routes import router as app_routes

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

app.register_blueprint(app_routes)

# Serve uploaded images
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# Ensure uploads folder exists
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

if __name__ == "__main__":
    app.run(debug=True)
