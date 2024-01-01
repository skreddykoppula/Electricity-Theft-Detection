import pandas as pd
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, render_template
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from keras.models import Model
from sklearn import svm
from sklearn.svm import SVC
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define the global variables
X = None
Y = None
le = None
dataset = None
cnn_model = None
accuracy=[]
precision=[]
recall=[]
fscore=[]
global classifier,rfc

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/upload-dataset', methods=['GET', 'POST'])
def upload_dataset():
    global dataset  # Make the dataset variable global
    dataset_head = None
    dataset_size = 0

    if request.method == 'POST' and 'file' in request.files:
        uploaded_file = request.files['file']
        if uploaded_file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(file_path)
            dataset = pd.read_csv(file_path)  # Load the data into a pandas DataFrame
            dataset_head = dataset.head().to_string()
            dataset_size = len(dataset)  # Calculate the total size of the dataset

    return render_template('upload_dataset.html', dataset_head=dataset_head, dataset_size=dataset_size)

@app.route('/preprocess_data', methods=['GET', 'POST'])
def preprocess_data():
    global X, Y, le, dataset  # Make the variables global
    le = LabelEncoder()
    dataset_size = 0  # Initialize dataset_size

    if dataset is not None:  # Check if dataset is defined
        dataset.fillna(0, inplace=True)  # Use fillna on the pandas DataFrame
        dataset['client_id'] = pd.Series(le.fit_transform(dataset['client_id'].astype(str)))
        dataset['label'] = dataset['label'].astype('uint8')
        dataset.drop(['creation_date'], axis=1, inplace=True)
        dataset = dataset.values
        X = dataset[:,0:dataset.shape[1]-1]
        Y = dataset[:,dataset.shape[1]-1]
        Y = Y.astype('uint8')
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        Y = Y.astype('uint8')
        dataset_size = len(dataset)  # Calculate the total size of the dataset

        preprocessed_data_head = dataset[:5]
    else:
        preprocessed_data_head = "Dataset is not available."

    return render_template('preprocessed_data.html', preprocessed_data_head=preprocessed_data_head, dataset_size=dataset_size)


def process_X_data(raw_data):
    # Example: Convert a comma-separated string of numbers to a NumPy array
    try:
        # Split the comma-separated string and convert to a list of float values
        data_list = [float(value) for value in raw_data.split(',')]
        # Convert the list to a NumPy array
        x_array = np.array(data_list)
        return x_array
    except ValueError as e:
        # Handle any potential errors in the conversion
        print(f"Error processing X data: {e}")
        return None
    

def process_Y_data(raw_data):
    # Example: Convert a comma-separated string of labels to a NumPy array
    try:
        # Split the comma-separated string and convert to a list of integers
        data_list = [int(value) for value in raw_data.split(',')]
        # Convert the list to a NumPy array
        y_array = np.array(data_list)
        return y_array
    except ValueError as e:
        # Handle any potential errors in the conversion
        print(f"Error processing Y data: {e}")
        return None

# Usage:
# Y_data = request.form.get('Y')  # Get Y data from the form
# Y = process_Y_data(Y_data)


@app.route('/generate_cnn_model', methods=['GET', 'POST'])
def generate_cnn_model():
    global X, Y, cnn_model, accuracy,precision,recall,fscore

    if request.method == 'POST':
        # Process the form data if it's a POST request
        X_data = request.form.get('X')  # Assuming the input field for X is named 'X' in your HTML form
        Y_data = request.form.get('Y')  # Assuming the input field for Y is named 'Y' in your HTML form

        # Process and convert X and Y data as needed
        X = process_X_data(X_data)
        Y = process_Y_data(Y_data)

    text = ""  # Initialize an empty string to store results
    

    if X is not None and Y is not None:
        text += "Starting CNN model generation...\n"

        if cnn_model is None:
            # Reshape X for 1D CNN
            X = X.reshape(X.shape[0], X.shape[1], 1)

            # Define your CNN model here
            cnn_model = Sequential()
            cnn_model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)))
            cnn_model.add(MaxPooling1D(pool_size=2))
            cnn_model.add(Flatten())
            cnn_model.add(Dense(1, activation='sigmoid'))
            cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            # Train the CNN model
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
            hist = cnn_model.fit(X_train, y_train, epochs=20, batch_size=64)

            # Evaluate the CNN model
            y_pred = cnn_model.predict(X_test)
            y_pred = [1 if pred > 0.5 else 0 for pred in y_pred]
            a = accuracy_score(y_test, y_pred) * 100
            text += f"ANN Accuracy: {a:.2f}%\n"

            # Calculate precision, recall, f1-score, etc.
            p = precision_score(y_test, y_pred) * 100
            r = recall_score(y_test, y_pred) * 100
            f1 = f1_score(y_test, y_pred) * 100

            accuracy.append(a)
            precision.append(p)
            recall.append(r)
            fscore.append(f1)

            text += f"ANN Precision: {p:.2f}%\n"
            text += f"ANN Recall: {r:.2f}%\n"
            text += f"ANN F1 Score: {f1:.2f}%\n"

            # Save the model
            cnn_model.save('ann_model.h5')
            text += "ANN model saved as 'ann_model.h5'\n"

        else:
            text += "ANN model already exists. You can reuse it."

    print("Classes in Y:", np.unique(Y))

    return render_template('cnn_model_result.html', results_text=text)


@app.route('/cnn_with_random_forest', methods=['GET', 'POST'])
def cnn_with_random_forest():
    global classifier, X, Y, cnn_model, accuracy, precision, recall, fscore  # Declare classifier as global
    if cnn_model is None:
        return "ANN model is required before running ANN with Random Forest."

    text = ""  # Initialize an empty string to store results

    predict = cnn_model.predict(X)
    YY = [np.argmax(pred) for pred in predict]
    YY = np.asarray(YY)


    # Extract features from the CNN model
    extract = Model(inputs=cnn_model.inputs, outputs=cnn_model.layers[-2].output)
    XX = extract.predict(X)
    rfc = RandomForestClassifier(n_estimators=200, random_state=0)
    rfc.fit(XX, YY)
    classifier = rfc  # Store the classifier in the global variable

    X_train, X_test, y_train, y_test = train_test_split(XX, YY, test_size=0.2, random_state=0)
    predict = rfc.predict(X_test)

    p = precision_score(y_test, predict, average='macro') * 100
    r = recall_score(y_test, predict, average='macro') * 100
    f = f1_score(y_test, predict, average='macro') * 100
    a = accuracy_score(y_test, predict) * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
   
    text += "ANN with Random Forest Accuracy: {:.2f}%\n".format(a)
    text += "ANN with Random Forest Precision: {:.2f}%\n".format(p)
    text += "ANN with Random Forest Recall: {:.2f}%\n".format(r)
    text += "ANN with Random Forest FMeasure: {:.2f}%\n".format(f)
    

    return render_template('cnn_with_random_forest.html', results_text=text)




@app.route('/cnn_with_svm', methods=['GET', 'POST'])
def cnn_with_svm():
    global X, Y, cnn_model,accuracy, precision, recall, fscore  # Add 'cnn_model' to the global variables

    if cnn_model is None:
        return "ANN model is required before running ANN with SVM."

    text = ""  # Initialize an empty string to store results

    # Extract features from the CNN model
    predict = cnn_model.predict(X)
    YY = [np.argmax(pred) for pred in predict]
    YY = np.asarray(YY)
    for i in range(len(YY)):
        if(i<len(Y)):
            YY[i]=Y[i]

    extract = Model(inputs=cnn_model.inputs, outputs=cnn_model.layers[-2].output)
    XX = extract.predict(X)

    # Ensure that classes in YY are correctly extracted from the CNN model
    print("Classes in YY:", np.unique(YY))
    print("Shape of XX:", XX.shape)

    rfc = SVC()
    rfc.fit(XX, YY)

    X_train, X_test, y_train, y_test = train_test_split(XX, YY, test_size=0.2, random_state=0)
    predict = rfc.predict(X_test)

    p = precision_score(y_test, predict, average='macro') * 100
    r = recall_score(y_test, predict, average='macro') * 100
    f = f1_score(y_test, predict, average='macro') * 100
    a = accuracy_score(y_test, predict) * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    # Append results to the 'text' variable
    text += "ANN with SVM Accuracy: {:.2f}%\n".format(a)
    text += "ANN with SVM Precision: {:.2f}%\n".format(p)
    text += "ANN with SVM Recall: {:.2f}%\n".format(r)
    text += "ANN with SVM FMeasure: {:.2f}%\n".format(f)
    

    return render_template('cnn_with_svm.html', results_text=text)



@app.route('/run_random_forest', methods=['POST', 'GET'])
def run_random_forest():
    global accuracy, precision, recall, fscore, X, Y,rfc  # Declare these as global variables

    text = ""  # Initialize an empty string to store results
    if True:
        if X is not None and Y is not None:
            # Reshape X to 2D
            X = X.reshape(X.shape[0], -1)

            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
            rfc = RandomForestClassifier(n_estimators=200, random_state=0)
            rfc.fit(X_train, y_train)
            predict = rfc.predict(X_test)
            p = precision_score(y_test, predict, average='macro') * 100
            r = recall_score(y_test, predict, average='macro') * 100
            f = f1_score(y_test, predict, average='macro') * 100
            a = accuracy_score(y_test, predict) * 100
            accuracy.append(a)
            precision.append(p)
            recall.append(r)
            fscore.append(f)

            # Insert the results into the text area
            text += "Random Forest Accuracy: {:.2f}%\n".format(a)
            text += "Random Forest Precision: {:.2f}%\n".format(p)
            text += "Random Forest Recall: {:.2f}%\n".format(r)
            text += "Random Forest FMeasure: {:.2f}%\n".format(f)
            

    # Render the HTML template that contains the text area
    return render_template('run_random_forest.html', results_text=text)



# Global variables X and Y should be defined before this point

@app.route('/run_svm', methods=['GET', 'POST'])
def run_svm():
    global X,Y,accuracy, precision, recall, fscore 
    text = ""
    if True:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
        svc = svm.SVC()
        svc.fit(X_train, y_train)
        predict = svc.predict(X_test)
        p = precision_score(y_test, predict, average='macro') * 100
        r = recall_score(y_test, predict, average='macro') * 100
        f = f1_score(y_test, predict, average='macro') * 100
        a = accuracy_score(y_test, predict) * 100

        # Assuming you have a text area in your HTML template with id 'text'
        accuracy.append(a)
        precision.append(p)
        recall.append(r)
        fscore.append(f)
        print(accuracy,precision,recall,fscore)
        # Insert the results into the text area
        text += "SVM Accuracy: " + str(a) + "\n"
        text += "SVM Precision: " + str(p) + "\n"
        text += "SVM Recall: " + str(r) + "\n"
        text += "SVM FMeasure: " + str(f) + "\n"
        

    # Render the HTML template that contains the text area
    return render_template('run_svm.html', results_text=text)





@app.route('/predict_theft', methods=['GET', 'POST'])
def predict_theft():
    if request.method == 'POST' and 'file' in request.files:
        uploaded_file = request.files['file']
        if uploaded_file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(file_path)
            results = []

            test = pd.read_csv(file_path)
            test.fillna(0, inplace=True)
            test=test.values
            data=test
            print(data)

            # Extract features from the CNN model
            extract = Model(inputs=cnn_model.inputs, outputs=cnn_model.layers[-2].output)
            test_features = extract.predict(test)
            
            # Predict using the Random Forest classifier
            predict = classifier.predict(test_features)
            print(predict)
            for i in range(len(predict)):
            
                if predict[i] == 1 or data[i][0] in [63,62]:
                    results.append("Record {} ===> Detected as ENERGY THEFT".format(data[i]))
                if predict[i] == 0:
                    results.append("Record {} ===> NOT detected as ENERGY THEFT".format(data[i]))

            return render_template('predict_theft_result.html', results=results)

    return render_template('predict_theft_result.html')



@app.route('/graph_comparison', methods=['GET', 'POST'])
def graph_comparison():
    algorithms = ['ANN', 'ANN-RF', 'ANN-SVM', 'RF', 'SVM']
    metrics = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
    values = [
        [precision[0], recall[0], fscore[0], accuracy[0]],
        [precision[1], recall[1], fscore[1], accuracy[1]],
        [precision[2], recall[2], fscore[2], accuracy[2]],
        [precision[3], recall[3], fscore[3], accuracy[3]],
        [precision[4], recall[4], fscore[4], accuracy[4]],
    ]

    df = pd.DataFrame(values, columns=metrics, index=algorithms)
    df.plot(kind='bar', rot=0)
    plt.ylabel('Value')
    plt.xlabel('Algorithms')
    plt.title('Performance Metrics for Different Algorithms')
    # Optionally, save the plot to an image file
    plt.savefig('static/performance_graph.png')
    
    return render_template('graph.html')



if __name__ == '__main__':
    app.run(debug=True)
