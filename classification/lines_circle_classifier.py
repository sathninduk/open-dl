import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
import pickle


class NeuralNetwork:
    def __init__(self, input_size=2, hidden_size=10, output_size=2):
        # Initialize weights with small random values
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, X, y, output, learning_rate=0.1):
        m = X.shape[0]

        # Output layer gradients
        dz2 = output - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Hidden layer gradients
        dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, epochs=1000, learning_rate=0.1):
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)


class PatternClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pattern Classifier - Neural Network Training")

        # Canvas settings
        self.canvas_width = 700
        self.canvas_height = 700

        # Data storage
        self.circle_points = []
        self.line_points = []
        self.nn = None
        self.is_trained = False

        # Phase tracking
        self.phase = "collecting_circles"  # collecting_circles, collecting_lines, testing
        self.circles_needed = 10
        self.lines_needed = 10

        # Current drawing path for testing phase
        self.current_path = []

        # Create UI
        self.create_ui()
        self.update_instruction()

    def create_ui(self):
        # Control frame
        control_frame = tk.Frame(self.root, bg="#2c3e50")
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=0, pady=0)

        # Load Model Button
        tk.Button(control_frame, text="Load Model", command=self.load_model,
                  bg="#3498db", fg="black", font=("Arial", 11, "bold"),
                  padx=15, pady=8).pack(side=tk.LEFT, padx=10, pady=10)

        # Clear/Reset Button
        tk.Button(control_frame, text="Start Over", command=self.reset_app,
                  bg="#e74c3c", fg="black", font=("Arial", 11, "bold"),
                  padx=15, pady=8).pack(side=tk.LEFT, padx=5, pady=10)

        # Save Model Button
        self.save_button = tk.Button(control_frame, text="Save Model", command=self.save_model,
                                     bg="#27ae60", fg="black", font=("Arial", 11, "bold"),
                                     padx=15, pady=8, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=5, pady=10)

        # Clear Drawing Button (for testing phase)
        self.clear_draw_button = tk.Button(control_frame, text="Clear Drawing",
                                           command=self.clear_current_drawing,
                                           bg="#f39c12", fg="black", font=("Arial", 11, "bold"),
                                           padx=15, pady=8, state=tk.DISABLED)
        self.clear_draw_button.pack(side=tk.LEFT, padx=5, pady=10)

        # Instruction label
        self.instruction_label = tk.Label(self.root, text="",
                                          font=("Arial", 14, "bold"),
                                          bg="#34495e", fg="black", pady=15)
        self.instruction_label.pack(side=tk.TOP, fill=tk.X)

        # Status label
        self.status_label = tk.Label(self.root, text="",
                                     font=("Arial", 11),
                                     bg="#ecf0f1", fg="#2c3e50", pady=10)
        self.status_label.pack(side=tk.TOP, fill=tk.X)

        # Canvas
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height,
                                bg="white", cursor="crosshair", highlightthickness=2,
                                highlightbackground="#34495e")
        self.canvas.pack(padx=10, pady=10)

        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def update_instruction(self):
        if self.phase == "collecting_circles":
            remaining = self.circles_needed - len(self.circle_points)
            self.instruction_label.config(text=f"📍 Draw CIRCLES - {remaining} more needed", bg="#e74c3c")
            self.status_label.config(text=f"Circles collected: {len(self.circle_points)}/{self.circles_needed}")
        elif self.phase == "collecting_lines":
            remaining = self.lines_needed - len(self.line_points)
            self.instruction_label.config(text=f"📍 Draw LINES - {remaining} more needed", bg="#3498db")
            self.status_label.config(text=f"Lines collected: {len(self.line_points)}/{self.lines_needed}")
        elif self.phase == "testing":
            self.instruction_label.config(text="✏️ Draw anything! The AI will classify it in real-time", bg="#27ae60")
            self.canvas.config(bg="black")
            self.clear_draw_button.config(state=tk.NORMAL)

    def on_mouse_down(self, event):
        if self.phase in ["collecting_circles", "collecting_lines"]:
            self.current_path = []

    def on_mouse_drag(self, event):
        x, y = event.x, event.y

        if self.phase == "collecting_circles" or self.phase == "collecting_lines":
            # Normalize coordinates
            norm_x = x / self.canvas_width
            norm_y = y / self.canvas_height

            # Add to current path
            self.current_path.append([norm_x, norm_y])

            # Draw on canvas
            if self.phase == "collecting_circles":
                color = "red"
            else:
                color = "blue"

            r = 3
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, outline=color)

        elif self.phase == "testing":
            # Real-time classification
            norm_x = x / self.canvas_width
            norm_y = y / self.canvas_height

            # Predict
            point = np.array([[norm_x, norm_y]])
            prediction = self.nn.predict(point)[0]
            output = self.nn.forward(point)

            circle_prob = output[0][0] * 100
            line_prob = output[0][1] * 100

            # Draw with color based on prediction
            if prediction == 0:
                color = "#ff4444"  # Red for circle
            else:
                color = "#4444ff"  # Blue for line

            r = 4
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, outline=color)

            # Update status with probabilities
            self.status_label.config(
                text=f"🔴 Circle: {circle_prob:.1f}%  |  🔵 Line: {line_prob:.1f}%  |  Predicted: {'CIRCLE' if prediction == 0 else 'LINE'}"
            )

    def on_mouse_up(self, event):
        if self.phase == "collecting_circles":
            if len(self.current_path) > 0:
                self.circle_points.extend(self.current_path)
                self.current_path = []

                if len(self.circle_points) >= self.circles_needed:
                    self.phase = "collecting_lines"
                    self.canvas.delete("all")
                    self.update_instruction()
                else:
                    self.update_instruction()

        elif self.phase == "collecting_lines":
            if len(self.current_path) > 0:
                self.line_points.extend(self.current_path)
                self.current_path = []

                if len(self.line_points) >= self.lines_needed:
                    # Start training
                    self.train_network()
                else:
                    self.update_instruction()

    def train_network(self):
        self.status_label.config(text="🔄 Training neural network... Please wait...")
        self.root.update()

        # Prepare training data
        X_circles = np.array(self.circle_points)
        X_lines = np.array(self.line_points)

        X = np.vstack([X_circles, X_lines])

        # Create labels (one-hot encoded)
        y_circles = np.zeros((len(X_circles), 2))
        y_circles[:, 0] = 1  # Class 0 for circles

        y_lines = np.zeros((len(X_lines), 2))
        y_lines[:, 1] = 1  # Class 1 for lines

        y = np.vstack([y_circles, y_lines])

        # Train neural network
        self.nn = NeuralNetwork(input_size=2, hidden_size=20, output_size=2)
        self.nn.train(X, y, epochs=2000, learning_rate=0.5)

        self.is_trained = True

        # Calculate accuracy
        predictions = self.nn.predict(X)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == true_labels) * 100

        # Move to testing phase
        self.phase = "testing"
        self.canvas.delete("all")
        self.update_instruction()
        self.save_button.config(state=tk.NORMAL)

        messagebox.showinfo("Training Complete!",
                            f"✅ Neural network trained successfully!\n\n"
                            f"Training Accuracy: {accuracy:.1f}%\n\n"
                            f"Now draw anything on the black canvas!\n"
                            f"The AI will classify it in real-time.")

    def clear_current_drawing(self):
        self.canvas.delete("all")
        self.status_label.config(text="Canvas cleared. Draw something new!")

    def save_model(self):
        if not self.is_trained:
            messagebox.showwarning("No Model", "Please train a model first!")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Model files", "*.pkl"), ("All files", "*.*")],
            title="Save Model"
        )

        if filename:
            model_data = {
                'weights': {
                    'W1': self.nn.W1,
                    'b1': self.nn.b1,
                    'W2': self.nn.W2,
                    'b2': self.nn.b2
                },
                'circle_points': self.circle_points,
                'line_points': self.line_points
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
                self.nn.W1 = model_data['weights']['W1']
                self.nn.b1 = model_data['weights']['b1']
                self.nn.W2 = model_data['weights']['W2']
                self.nn.b2 = model_data['weights']['b2']

                # Restore training points
                self.circle_points = model_data['circle_points']
                self.line_points = model_data['line_points']

                self.is_trained = True

                # Move directly to testing phase
                self.phase = "testing"
                self.canvas.delete("all")
                self.update_instruction()
                self.save_button.config(state=tk.NORMAL)

                messagebox.showinfo("Success",
                                    f"✅ Model loaded successfully!\n\n"
                                    f"You can now draw on the black canvas.\n"
                                    f"The AI will classify your drawings in real-time!")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model:\n\n{str(e)}")

    def reset_app(self):
        if self.is_trained:
            confirm = messagebox.askyesno("Confirm Reset",
                                          "Are you sure you want to start over?\n\n"
                                          "This will clear all data and reset the application.")
            if not confirm:
                return

        self.canvas.delete("all")
        self.canvas.config(bg="white")
        self.circle_points = []
        self.line_points = []
        self.current_path = []
        self.is_trained = False
        self.nn = None
        self.phase = "collecting_circles"
        self.save_button.config(state=tk.DISABLED)
        self.clear_draw_button.config(state=tk.DISABLED)
        self.update_instruction()


if __name__ == "__main__":
    root = tk.Tk()
    app = PatternClassifierApp(root)
    root.mainloop()