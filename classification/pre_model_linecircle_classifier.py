import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
import pickle
from PIL import Image, ImageDraw
import gzip
import os


class NeuralNetwork:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        # Initialize weights with Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, X, y, output, learning_rate=0.01):
        m = X.shape[0]

        # Output layer gradients
        dz2 = output - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Hidden layer gradients
        dz1 = np.dot(dz2, self.W2.T) * self.relu_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, epochs=10, batch_size=128, learning_rate=0.01):
        n_samples = X.shape[0]

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, output, learning_rate)

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)


def download_mnist():
    """Verify MNIST files exist in ./mnist_data and return that directory.

    NOTE: This function will NOT download files. If files are missing, it raises FileNotFoundError
    with instructions to obtain them from the original MNIST hosting site.
    """
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }

    data_dir = 'mnist_data'

    # Directory must already exist and contain the .gz files
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"MNIST data directory '{data_dir}' not found. Please create it and place the MNIST .gz files there."
        )

    missing = []
    for filename in files.values():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            missing.append(filename)

    if missing:
        raise FileNotFoundError(
            f"Missing MNIST files in '{data_dir}': {', '.join(missing)}.\n"
            f"Please download them from http://yann.lecun.com/exdb/mnist/ and place them into the folder."
        )

    # All required files are present
    return data_dir


def load_mnist_images(filepath):
    """Load MNIST images from gz file"""
    with gzip.open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)


def load_mnist_labels(filepath):
    """Load MNIST labels from gz file"""
    with gzip.open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data


class MNISTClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Handwritten Digit Classifier")

        # Canvas settings
        self.canvas_size = 280  # 28x28 scaled up by 10
        self.brush_size = 20

        # Neural network
        self.nn = None
        self.is_trained = False

        # Drawing
        self.image = Image.new('L', (28, 28), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None

        # Create UI
        self.create_ui()

        # Check for saved model
        self.check_saved_model()

    def create_ui(self):
        # Control frame
        control_frame = tk.Frame(self.root, bg="#2c3e50")
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=0, pady=0)

        # Train Button
        tk.Button(control_frame, text="Train", command=self.train_model,
                  bg="#e74c3c", fg="black", font=("Arial", 11, "bold"),
                  padx=15, pady=8).pack(side=tk.LEFT, padx=10, pady=10)

        # Load Model Button
        tk.Button(control_frame, text="Load Model", command=self.load_model,
                  bg="#3498db", fg="black", font=("Arial", 11, "bold"),
                  padx=15, pady=8).pack(side=tk.LEFT, padx=5, pady=10)

        # Save Model Button
        tk.Button(control_frame, text="Save Model", command=self.save_model,
                  bg="#27ae60", fg="black", font=("Arial", 11, "bold"),
                  padx=15, pady=8).pack(side=tk.LEFT, padx=5, pady=10)

        # Clear Button
        tk.Button(control_frame, text="Clear", command=self.clear_canvas,
                  bg="#f39c12", fg="black", font=("Arial", 11, "bold"),
                  padx=15, pady=8).pack(side=tk.LEFT, padx=5, pady=10)

        # Instruction label
        self.instruction_label = tk.Label(self.root,
                                          text="✏️ Draw a digit (0-9) and see the AI classify it!",
                                          font=("Arial", 14, "bold"),
                                          bg="white", fg="black", pady=15)
        self.instruction_label.pack(side=tk.TOP, fill=tk.X)

        # Status label
        self.status_label = tk.Label(self.root, text="Load or train a model to begin",
                                     font=("Arial", 12, "bold"),
                                     bg="#ecf0f1", fg="#2c3e50", pady=12)
        self.status_label.pack(side=tk.TOP, fill=tk.X)

        # Main container
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=20, pady=20)

        # Canvas
        canvas_frame = tk.Frame(main_frame)
        canvas_frame.pack(side=tk.LEFT, padx=10)

        tk.Label(canvas_frame, text="Draw Here:", font=("Arial", 12, "bold")).pack()

        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_size, height=self.canvas_size,
                                bg="black", cursor="crosshair", highlightthickness=2,
                                highlightbackground="#34495e")
        self.canvas.pack()

        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # Prediction display
        prediction_frame = tk.Frame(main_frame, bg="#000", relief=tk.RAISED, borderwidth=2)
        prediction_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH)

        tk.Label(prediction_frame, text="Predictions:", font=("Arial", 14, "bold"),
                 bg="#000").pack(pady=10)

        self.prediction_labels = []
        for i in range(10):
            frame = tk.Frame(prediction_frame, bg="#000")
            frame.pack(fill=tk.X, padx=10, pady=2)

            digit_label = tk.Label(frame, text=f"{i}:", font=("Arial", 12, "bold"),
                                   width=2, bg="#000")
            digit_label.pack(side=tk.LEFT)

            bar_canvas = tk.Canvas(frame, width=200, height=20, bg="white",
                                   highlightthickness=1, highlightbackground="#95a5a6")
            bar_canvas.pack(side=tk.LEFT, padx=5)

            prob_label = tk.Label(frame, text="0.0%", font=("Arial", 10),
                                  width=6, bg="#000")
            prob_label.pack(side=tk.LEFT)

            self.prediction_labels.append((bar_canvas, prob_label))

    def on_mouse_down(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def on_mouse_drag(self, event):
        if self.last_x and self.last_y:
            # Draw on canvas
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                    fill="white", width=self.brush_size,
                                    capstyle=tk.ROUND, smooth=True)

            # Draw on PIL image (scaled down)
            x1 = self.last_x * 28 // self.canvas_size
            y1 = self.last_y * 28 // self.canvas_size
            x2 = event.x * 28 // self.canvas_size
            y2 = event.y * 28 // self.canvas_size

            self.draw.line([x1, y1, x2, y2], fill=255, width=2)

            self.last_x = event.x
            self.last_y = event.y

            # Classify after drawing
            if self.is_trained:
                self.classify_digit()

    def on_mouse_up(self, event):
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new('L', (28, 28), 0)
        self.draw = ImageDraw.Draw(self.image)

        # Clear predictions
        for bar_canvas, prob_label in self.prediction_labels:
            bar_canvas.delete("all")
            prob_label.config(text="0.0%")

        if self.is_trained:
            self.status_label.config(text="Canvas cleared. Draw a digit!")

    def classify_digit(self):
        if not self.is_trained:
            return

        # Convert image to numpy array and normalize
        img_array = np.array(self.image).flatten() / 255.0
        img_array = img_array.reshape(1, -1)

        # Get predictions
        output = self.nn.forward(img_array)
        probabilities = output[0] * 100
        prediction = np.argmax(probabilities)

        # Update display
        self.status_label.config(text=f"Predicted Digit: {prediction} (Confidence: {probabilities[prediction]:.1f}%)")

        # Update probability bars
        max_prob = np.max(probabilities)
        for i, (bar_canvas, prob_label) in enumerate(self.prediction_labels):
            bar_canvas.delete("all")
            prob = probabilities[i]

            # Draw probability bar
            bar_width = int((prob / 100) * 200)
            color = "#27ae60" if i == prediction else "#3498db"
            bar_canvas.create_rectangle(0, 0, bar_width, 20, fill=color, outline="")

            # Update probability text
            prob_label.config(text=f"{prob:.1f}%",
                              font=("Arial", 10, "bold" if i == prediction else "normal"))

    def train_model(self):
        confirm = messagebox.askyesno(
            "Train",
            "This will train a neural network using the MNIST files located in ./mnist_data.\n\n"
            "Make sure the files (train-images-idx3-ubyte.gz, train-labels-idx1-ubyte.gz,\n"
            "t10k-images-idx3-ubyte.gz, t10k-labels-idx1-ubyte.gz) are present in that folder.\n\n"
            "Continue?"
        )

        if not confirm:
            return

        # Ensure accuracy is defined even if training aborts early (silences static analysis warnings)
        accuracy = 0.0

        try:
            self.status_label.config(text="📂 Checking MNIST dataset...")
            self.root.update()

            # Verify dataset exists (no downloading)
            data_dir = download_mnist()

            self.status_label.config(text="📂 Loading dataset...")
            self.root.update()

            # Load data
            train_images = load_mnist_images(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
            train_labels = load_mnist_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))

            # Prepare data
            X_train = train_images.reshape(-1, 784) / 255.0  # Normalize

            # One-hot encode labels
            y_train = np.zeros((len(train_labels), 10))
            y_train[np.arange(len(train_labels)), train_labels] = 1

            # Use subset for faster training
            n_samples = 10000  # Use 10k samples for faster training
            X_train = X_train[:n_samples]
            y_train = y_train[:n_samples]

            self.status_label.config(text=f"🔄 Training neural network on {n_samples} samples...")
            self.root.update()

            # Train
            self.nn = NeuralNetwork(input_size=784, hidden_size=128, output_size=10)

            for epoch in range(10):
                self.nn.train(X_train, y_train, epochs=1, batch_size=128, learning_rate=0.1)

                # Calculate accuracy
                predictions = self.nn.predict(X_train)
                true_labels = np.argmax(y_train, axis=1)
                accuracy = np.mean(predictions == true_labels) * 100

                self.status_label.config(text=f"🔄 Epoch {epoch + 1}/10 - Accuracy: {accuracy:.1f}%")
                self.root.update()

            self.is_trained = True

            messagebox.showinfo(
                "Training Complete!",
                f"✅ Neural network trained successfully!\n\n"
                f"Final Accuracy: {accuracy:.1f}%\n"
                f"Training samples: {n_samples}\n\n"
                f"Draw a digit to test the classifier!"
            )

            self.status_label.config(text=f"✅ Model trained! Accuracy: {accuracy:.1f}% - Draw a digit!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model:\n\n{str(e)}")
            self.status_label.config(text="❌ Training failed. Try again or load a model.")

    def save_model(self):
        if not self.is_trained:
            messagebox.showwarning("No Model", "Please train or load a model first!")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Model files", "*.pkl"), ("All files", "*.*")],
            title="Save Model"
        )

        if filename:
            model_data = {
                'W1': self.nn.W1,
                'b1': self.nn.b1,
                'W2': self.nn.W2,
                'b2': self.nn.b2
            }

            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)

            messagebox.showinfo("Success", f"✅ Model saved successfully!\n\n{filename}")

    def load_model(self):
        filename = filedialog.askopenfilename(
            filetypes=[("Model files", "*.pkl"), ("All files", "*.*")],
            title="Load Model"
        )

        if filename:
            try:
                with open(filename, 'rb') as f:
                    model_data = pickle.load(f)

                # Restore neural network
                self.nn = NeuralNetwork()
                self.nn.W1 = model_data['W1']
                self.nn.b1 = model_data['b1']
                self.nn.W2 = model_data['W2']
                self.nn.b2 = model_data['b2']

                self.is_trained = True
                self.clear_canvas()

                self.status_label.config(text="✅ Model loaded! Draw a digit to classify it!")

                messagebox.showinfo("Success",
                                    "✅ Model loaded successfully!\n\n"
                                    "Draw a digit (0-9) on the canvas!")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model:\n\n{str(e)}")

    def check_saved_model(self):
        """Check if mnist_model.pkl exists in current directory"""
        if os.path.exists('mnist_model.pkl'):
            load = messagebox.askyesno(
                "Saved Model Found",
                "Found a saved MNIST model (mnist_model.pkl).\n\n"
                "Would you like to load it?"
            )
            if load:
                try:
                    with open('mnist_model.pkl', 'rb') as f:
                        model_data = pickle.load(f)

                    self.nn = NeuralNetwork()
                    self.nn.W1 = model_data['W1']
                    self.nn.b1 = model_data['b1']
                    self.nn.W2 = model_data['W2']
                    self.nn.b2 = model_data['b2']

                    self.is_trained = True
                    self.status_label.config(text="✅ Model loaded! Draw a digit to classify it!")
                except:
                    pass


if __name__ == "__main__":
    root = tk.Tk()
    app = MNISTClassifierApp(root)
    root.mainloop()