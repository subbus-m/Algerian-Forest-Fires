import json
import pickle
from flask import Flask, request, app, jsonify, render_template
import mysql.connector as conn
import numpy as np

app = Flask(__name__)
# Load Pickle files
class_model = pickle.load(open('rfclmodel.pkl', 'rb'))
reg_model = pickle.load(open('rfregmodel.pkl', 'rb'))

# home just to render the home template. This will be triggered when only IP is given without any API name.
# methods = ['POST'] is not given here because it is needed only for APIs. This is just a route which is
# called from html page which is inside application. If this is supposed to get called from outside
# application, then methods = ['POST'] needs to be placed.
@app.route('/')
def home():
    return render_template('home.html')

# Basic API to predict class based on input values. We can call it from POSTMAN or any other application.
# here methods = ['POST'] is given since this procedure 'predict_file' is used as an API which can be
# called from any application.
@app.route('/predict_fire', methods = ['POST'])
def predict_fire():
    data = request.json['data']             # POST is to get the values in form or seperately
    new_data = [list(data.values())]        # GET is to receive the parameter values in url itself
    output = class_model.predict(new_data)[0]
    return jsonify(output)

# Basic API to predict temperature based on input values. We can call it from POSTMAN or any other application.
@app.route('/predict_temperature', methods = ['POST'])
def predict_temperature():
    data = request.json['data']
    new_data = [list(data.values())]
    output = reg_model.predict(new_data)
    return jsonify(str(output))

# When href link of predict class is clicked on home.html, this will be triggered.
# This method is just to render the template (blank page of class.html) asking for the data.
# Since we are just rendering the blank template without data we don't need methods=['POST']
# If we are receiving the data and processing it needs an API like 'predict_class' and 'predict_temperature'
@app.route('/classp')
def classp():    # Function to predict the values
    return render_template('class.html')

# This API receives the data received as input from temprature.html in the above render in tempp and process
# the data and gives final prediction. Since we are having logic of handling data received from html page,
# we are making it as API by mentioning methods=['POST']. Even in class.html page, we mention method='post'
# in action so that this predict_class API will be triggered when 'predict class' button is clicked in
# class.html page.
@app.route('/predict_class_html', methods = ['POST'])
def predict_class_html():    # Function to predict the values

    data = [float(x) for x in request.form.values()]    # request.form.values will retrieve values from the web form we created
    final_features = [np.array(data)]
    output = class_model.predict(final_features)[0]
    return render_template('class.html', prediction_text = 'The predicted class is - {}'.format(output))

# When href link of predict temperature is clicked on home.html, this will be triggered.
# This method is just to render the template (blank page of class.html) asking for the data.
@app.route('/tempp')
def tempp():    # Function to predict the values
    return render_template('temprature.html')

# This API receives the data received as input from temprature.html in the above render in tempp and process
# the data and gives final prediction. Since we are having logic of handling data received from html page,
# we are making it as API by mentioning methods=['POST']. Even in class.html page, we mention method='post'
# in action so that this predict_class API will be triggered when 'predict class' button is clicked in
# class.html page.
@app.route('/predict_temp_html', methods = ['POST'])
def predict_temp_html():    # Function to predict the values

    data = [float(x) for x in request.form.values()]    # request.form.values will retrieve values from the web form we created
    final_features = [np.array(data)]
    output = reg_model.predict(final_features)[0]
    return render_template('temprature.html', prediction_text = 'The predicted temperature is - {}'.format(output))

# This API is to predict class in batch input from the mysql table algeriaff
# When this API is called, it will read mysql table algeriaff and predicts class using model and provide
# batch output for all the input from table.
@app.route('/predict_class_mysql')
def predict_class_mysql():
    mydb = conn.Connect(host='localhost', user='root', passwd='mysql')
    cursor = mydb.cursor()
    cursor.execute('use project')
    cursor.execute('select day, month, year, temperature, rh, ws, rain, ffmc, dmc, dc, isi, bui, fwi from algeriaff')
    new_data = cursor.fetchall()
    output = class_model.predict(new_data)
    return render_template('home.html', prediction_text='The predicted class is - {}'.format(output))

if __name__ == '__main__':
    app.run(debug=True)
