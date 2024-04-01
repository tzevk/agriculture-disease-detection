from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from pymongo import MongoClient
from flask import Flask, render_template, request, redirect, url_for, session
from pymongo import MongoClient
import numpy as np
from PIL import Image
from skimage import filters, measure
import base64
import io
import matplotlib.pyplot as plt
from datetime import datetime
from skimage.morphology import binary_erosion, binary_dilation
from skimage import filters, measure

app = Flask(__name__)

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["crop_database"]
collection = db['crop_images']
users_collection = db["users"]
user_thoughts_collection = db["user_thoughts"]

app.secret_key = '1234'

# Route to handle chatbot requests
@app.route('/chatbot', methods=['POST'])
def chatbot():
    # Get user input from the request
    user_input = request.form['user_input']

    # Process user input and perform MongoDB queries
    response = process_query(user_input)

    # Return response to the chatbot interface
    return jsonify({'response': response})

# Function to process user input and perform MongoDB queries
def process_query(user_input):
    # Example: Detect query type based on keywords in user input
    if 'crop' in user_input:
        # Perform MongoDB query to search for crop information
        result = collection.find_one({'name': user_input})
        if result:
            # Return relevant information about the crop
            response = f"Here's what I found about {user_input}: {result}"
        else:
            response = f"Sorry, I couldn't find information about {user_input}."
    elif 'region' in user_input:
        # Perform MongoDB query to search for crop information by region
        results = collection.find({'region': user_input})
        if results:
            # Return relevant information about crops in the region
            response = f"Here are the crops in {user_input}: {results}"
        else:
            response = f"Sorry, I couldn't find information about crops in {user_input}."
    # Add more detection logic and MongoDB queries for other types of queries
    else:
        response = "Sorry, I couldn't understand your query. Please try again."

    return response

def edge_detection(image):
    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Step 1: Convert the image to grayscale
    gray = np.dot(image_array[..., :3], [0.299, 0.587, 0.114])

    # Display the grayscale image
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 3, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')

    # Step 2: Calculate the gradient in the x and y directions
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    grad_x = np.zeros_like(gray)
    grad_y = np.zeros_like(gray)

    # Iterate over the image and apply the Sobel kernels
    for i in range(1, gray.shape[0] - 1):
        for j in range(1, gray.shape[1] - 1):
            grad_x[i, j] = np.sum(kernel_x * gray[i - 1:i + 2, j - 1:j + 2])
            grad_y[i, j] = np.sum(kernel_y * gray[i - 1:i + 2, j - 1:j + 2])

    # Display the gradients in x and y directions
    plt.subplot(2, 3, 2)
    plt.imshow(grad_x, cmap='gray')
    plt.title('Gradient in X Direction')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(grad_y, cmap='gray')
    plt.title('Gradient in Y Direction')
    plt.axis('off')

    # Step 3: Calculate the magnitude of the gradient
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Display the gradient magnitude
    plt.subplot(2, 3, 4)
    plt.imshow(grad_mag, cmap='gray')
    plt.title('Gradient Magnitude')
    plt.axis('off')

    # Step 4: Apply a threshold to identify edges
    threshold = np.mean(grad_mag)
    edges = grad_mag > threshold

    # Display the edge image
    plt.subplot(2, 3, 5)
    plt.imshow(edges, cmap='gray')
    plt.title('Edge Image')
    plt.axis('off')
    kernel = np.ones((3, 3), dtype=np.uint8)
    edges = binary_erosion(edges, kernel)
    edges = binary_dilation(edges, kernel)

    plt.tight_layout()
    plt.show()

    return edges

def image_segmentation(image):
    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Convert the image to grayscale
    gray = np.average(image_array[..., :3], weights=[0.299, 0.587, 0.114], axis=-1)

    # Apply Gaussian blur to reduce noise
    blurred = filters.gaussian(gray, sigma=1)

    # Apply auto-thresholding to segment the image
    auto_thresh = filters.threshold_otsu(blurred)
    segmented = blurred > auto_thresh

    # Convert the segmented image to a PIL Image
    segmented_image = Image.fromarray((segmented * 255).astype(np.uint8))

    return segmented_image

@app.route('/recent_uploads', methods=['GET'])
def recent_uploads():
    search_query = request.args.get('search_query')
    if search_query:
        # Perform a search query in the database based on the search query
        uploads = collection.find({'name': {'$regex': f'.*{search_query}.*', '$options': 'i'}})
    else:
        # Fetch all recent uploads from the database
        uploads = collection.find()

    return render_template('recent_uploads.html', uploads=uploads)

@app.route('/results', methods=['GET', 'POST'])
def results():
    if request.method == 'POST':
        # Get the search query from the form
        search_query = request.form['search_query']

        # Search for images in the database
        result = collection.find_one({'name': search_query})

        if result:
            # Process the image if found
            img_bytes = result['image']
            img = Image.open(io.BytesIO(img_bytes))

            # Perform image processing (e.g., edge detection, segmentation)
            edges = edge_detection(img)
            segmented_image = image_segmentation(img)

            # Convert processed images to base64 for displaying in HTML
            buffer = io.BytesIO()
            img_edges = Image.fromarray((edges * 255).astype(np.uint8))
            img_edges.save(buffer, format='PNG')
            img_edges_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            buffer = io.BytesIO()
            segmented_image.save(buffer, format='PNG')
            segmented_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            return render_template('results.html', 
                                   edges=img_edges_base64, 
                                   segmented_image=segmented_image_base64)
        else:
            # Display error message if image not found
            return render_template('results.html', error='Invalid name')

    return render_template('results.html')

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users_collection.find_one({'username': username, 'password': password})
        if user:
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return 'Invalid username or password'
    return render_template('login.html')


def get_user_thoughts(username):
    # Fetch user thoughts from the database based on the username
    user = users_collection.find_one({'username': username})
    if user:
        return user.get('thoughts', '')  # Return user thoughts if found, otherwise an empty string
    else:
        return ''  # Return empty string if user is not found

@app.route('/user_page', methods=['GET', 'POST'])
def user_page():
    if 'username' in session:
        username = session['username']
        user = users_collection.find_one({'username': username})
        if user:
            if request.method == 'POST':
                new_thought = request.form.get('new_thought')  # Get the new thought from the form
                if new_thought:  # Check if the new thought is not empty
                    # Insert the new thought into the user_thoughts_collection
                    user_thoughts_collection.insert_one({'username': username, 'thought': new_thought})
                    # Redirect to the user page to prevent form resubmission on page refresh
                    return redirect(url_for('user_page'))

            user_thoughts_cursor = user_thoughts_collection.find({'username': username})  # Get user thoughts from the collection
            user_thoughts = [thought['thought'] for thought in user_thoughts_cursor]  # Extract thoughts from cursor
            return render_template('user_page.html', username=username, user_thoughts=user_thoughts, user=user)
        else:
            return "User not found"
    return redirect(url_for('/login'))

@app.route('/signup', methods=['POST'])
def signup():
    fullname = request.form['fullname']
    email = request.form['email']
    username = request.form['username']
    password = request.form['password']
    confirm_password = request.form['confirm_password']

    if users_collection.find_one({'username': username}):
        return 'Username already exists'

    if password != confirm_password:
        return 'Passwords do not match'

    users_collection.insert_one({
        'fullname': fullname,
        'email': email,
        'username': username,
        'password': password
    })

    return redirect('/login')

@app.route('/home')
def home():
    if 'username' in session:
        return render_template('home.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if request.method == 'POST':
        name = request.form['name']
        region = request.form['region']
        crop_type = request.form['crop_type']
        number = request.form['number']
        problem = request.form['problem']
        timestamp = datetime.now()

        # Save image to MongoDB
        img_file = request.files['image']
        img_bytes = img_file.read()
        img = Image.open(io.BytesIO(img_bytes))

        # Perform image processing (e.g., edge detection, segmentation)
        edges = edge_detection(img)
        segmented_image = image_segmentation(img)

        # Convert processed images to base64 for displaying in HTML
        buffer = io.BytesIO()
        img_edges = Image.fromarray(edges.astype('uint8') * 255)
        img_edges.save(buffer, format='PNG')
        img_edges_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        buffer = io.BytesIO()
        segmented_image.save(buffer, format='PNG')
        segmented_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Convert images to numpy arrays for storing in MongoDB
        image_array = np.array(img)
        segmented_array = np.array(segmented_image)

        # Insert data into MongoDB
        collection.insert_one({
            'name': name,
            'timestamp': timestamp,
            'region': region,
            'crop_type': crop_type,
            'number': number,
            'problem': problem,
            'image': img_bytes  # Store raw image bytes
        })

        # Pass processed images to HTML template for display
        return render_template('home.html', edges=img_edges_base64, segmented_image=segmented_image_base64)

    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)
