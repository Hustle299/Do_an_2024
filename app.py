from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import mysql.connector
from werkzeug.utils import secure_filename
from datetime import datetime

# Flask setup
app = Flask(__name__)
app.secret_key = "your_secret_key"
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
    if 'user_id' in session:
        return redirect(url_for('library'))
    return redirect(url_for('login'))

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

            cursor = db.cursor()
            cursor.execute(
                "INSERT INTO images (user_id, filename, upload_time) VALUES (%s, %s, %s)",
                (session['user_id'], filename, datetime.now())
            )
            db.commit()
            flash("Image uploaded successfully")
            return redirect(url_for('library'))
    return render_template('upload.html')

@app.route('/library')
def library():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM images WHERE user_id = %s", (session['user_id'],))
    images = cursor.fetchall()
    return render_template('library.html', images=images)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(debug=True)
