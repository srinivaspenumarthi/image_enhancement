from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import sqlite3
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model
model = tf.keras.models.load_model('underwater_Enhanced.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError()})

def enhance_image(image_path):
    # Open and preprocess the image
    image = Image.open(image_path).resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Enhance the image using the model
    enhanced_image = model.predict(image)[0]
    enhanced_image = (enhanced_image * 255).astype(np.uint8)
    enhanced_image = Image.fromarray(enhanced_image)

    # Save the enhanced image
    enhanced_filename = 'enhanced.png'
    enhanced_path = os.path.join(app.config['UPLOAD_FOLDER'], enhanced_filename)
    enhanced_image.save(enhanced_path)

    # Save the original image for comparison
    original_filename = 'original.png'
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
    Image.open(image_path).save(original_path)

    return enhanced_filename, original_filename

def visualize_results(x_array, decoded_imgs, save_path):
    plt.figure(figsize=(10, 5))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    # Original image
    ax = plt.subplot(1, 2, 1)
    plt.imshow(x_array[0])
    plt.title("Original")
    plt.axis("off")

    # Reconstructed image
    ax = plt.subplot(1, 2, 2)
    plt.imshow(decoded_imgs[0])
    plt.title("Reconstructed")
    plt.axis("off")

    plt.savefig(save_path)  # Save the figure
    plt.close()  # Close the figure to free memory

# Database setup
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (id INTEGER PRIMARY KEY, name TEXT, username TEXT UNIQUE, password TEXT, email TEXT, address TEXT)''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        address = request.form['address']
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("INSERT INTO users (name, username, password, email, address) VALUES (?, ?, ?, ?, ?)",
                      (name, username, password, email, address))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return "Username already exists. Please choose a different username."
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = c.fetchone()
        conn.close()
        if user:
            session['username'] = username
            return redirect(url_for('upload'))
        else:
            return "Invalid username or password."
    return render_template('login.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files['file']
        if file:
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Preprocess the image
                image = Image.open(filepath).resize((128, 128))
                x_array = np.array(image) / 255.0
                x_array = np.expand_dims(x_array, axis=0)

                # Predict the reconstructed image
                decoded_imgs = model.predict(x_array)

                # Save the visualization
                visualization_path = os.path.join(app.config['UPLOAD_FOLDER'], 'visualization.png')
                visualize_results(x_array, decoded_imgs, visualization_path)

                return render_template('result.html', visualization_image='visualization.png')
            except Exception as e:
                return f"An error occurred: {str(e)}"
    return render_template('upload.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)