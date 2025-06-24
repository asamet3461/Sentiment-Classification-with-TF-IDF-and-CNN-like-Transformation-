import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from scipy.signal import convolve2d


class TextClassifier:
    def __init__(self, data_path):
        """
        Initializes the TextClassifier by loading the dataset and performing necessary preprocessing.

        :param data_path: The path to the CSV dataset
        """
        # Read the dataset and remove any rows with missing values
        self.df = pd.read_csv(data_path)
        self.df.dropna(inplace=True)

        # Initialize the vectorizer and label encoder
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.encoder = LabelEncoder()

        # Transform the comments into TF-IDF features and encode the sentiment labels
        self.X = self.vectorizer.fit_transform(self.df["Comment"]).toarray()
        self.y = self.encoder.fit_transform(self.df["Sentiment"])

        # Split the dataset into training and test sets (80% training, 20% testing)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=42)

        # Define the convolution kernel
        self.kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

    def apply_convolution(self, X):
        """
        Applies the convolution operation to the input data.

        :param X: The input feature data
        :return: The convolved feature data
        """
        output = []
        for sample in X:
            # Reshape the input sample into a 2D matrix with 50 rows
            sample_2d = sample.reshape(50, -1)
            # Apply the convolution operation using the defined kernel
            conv_result = convolve2d(sample_2d, self.kernel, mode='valid')
            output.append(conv_result.flatten())
        return np.array(output)

    def train_model(self):
        """
        Trains a logistic regression model on the transformed features and evaluates its performance.

        :return: None
        """
        # Apply convolution to both training and test sets
        X_train_conv = self.apply_convolution(self.X_train)
        X_test_conv = self.apply_convolution(self.X_test)

        # Initialize and train the logistic regression model
        model = LogisticRegression(max_iter=200)
        model.fit(X_train_conv, self.y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test_conv)
        # Calculate and print the accuracy of the model
        accuracy = accuracy_score(self.y_test, y_pred)

        print(f"Test Accuracy: {accuracy:.4f}")


# Main execution block
if __name__ == "__main__":
    # Create an instance of the TextClassifier class and train the model
    classifier = TextClassifier("YoutubeCommentsDataSet.csv")
    classifier.train_model()
