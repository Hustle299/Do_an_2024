from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import mysql.connector
from werkzeug.utils import secure_filename
from datetime import datetime
from hand_measure.hand_measure import measure_hand

# Flask setup
app = Flask(__name__)
app.secret_key = "your_secret_key"
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CALIBRATION_FOLDER'] = os.path.join(app.static_folder, 'calibration')

# MySQL connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="duytung299",
    database="hand_measure_db"
)

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html', title="Home")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cursor = db.cursor()
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            flash("Username already exists")
            return redirect(url_for('register'))
        cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
        db.commit()
        flash("Registration successful. Please log in.")
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cursor = db.cursor()
        cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
        user = cursor.fetchone()
        if user:
            session['user_id'] = user[0]
            session['username'] = user[1]
            return redirect(url_for('library'))
        flash("Invalid credentials")
    return render_template('login.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        if 'image' not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash("No selected file")
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Call measure_hand to process the image
            measurements = measure_hand(filepath)

            # Store image and measurements in the database
            cursor = db.cursor()
            cursor.execute(
                "INSERT INTO images (user_id, filename, upload_time, measurements) VALUES (%s, %s, %s, %s)",
                (session['user_id'], filename, datetime.now(), str(measurements))
            )
            db.commit()
            flash("Image uploaded and measurements processed successfully")
            return redirect(url_for('library'))
    return render_template('upload.html')

@app.route('/library')
def library():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM images WHERE user_id = %s", (session['user_id'],))
    images = cursor.fetchall()
    for image in images:
        image['measurements'] = eval(image['measurements'])  # Convert string back to dictionary
    return render_template('library.html', images=images)

@app.route('/delete/<int:image_id>', methods=['POST'])
def delete_image(image_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    cursor = db.cursor()
    cursor.execute("SELECT filename FROM images WHERE id = %s AND user_id = %s", (image_id, session['user_id']))
    image = cursor.fetchone()

    if image:
        filename = image[0]  # Access the first element of the tuple
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(filepath):
            os.remove(filepath)  # Remove the file from the file system

        cursor.execute("DELETE FROM images WHERE id = %s", (image_id,))
        db.commit()
        flash("Image deleted successfully.")
    else:
        flash("Image not found or unauthorized access.")
    
    return redirect(url_for('library'))

@app.route('/calibration_upload', methods=['GET', 'POST'])
def calibration_upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part.')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file.')
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['CALIBRATION_FOLDER'], filename)
            file.save(upload_path)
            flash('Calibration image uploaded successfully.')
            return redirect(url_for('calibration_display'))

    return render_template('calibration_upload.html')

@app.route('/calibration_display', methods=['GET'])
def calibration_display():
    calibration_folder = app.config['CALIBRATION_FOLDER']
    calibration_images = os.listdir(calibration_folder)
    return render_template('calibration_display.html', calibration_images=calibration_images)

@app.route('/delete_calibration/<image_name>', methods=['POST'])
def delete_calibration(image_name):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    calibration_folder = app.config['CALIBRATION_FOLDER']
    filepath = os.path.join(calibration_folder, image_name)

    if os.path.exists(filepath):
        os.remove(filepath)  # Delete the image file
        flash('Calibration image deleted successfully.')
    else:
        flash('Image not found or unauthorized access.')

    return redirect(url_for('calibration_display'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
