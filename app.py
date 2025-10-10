from flask import Flask, render_template, request,send_file
import pandas as pd
import os
from datetime import datetime
from Easy_Visa.pipeline.prediction_pipeline import USvisaData, USvisaClassifier
from Easy_Visa.logging.logger import logger

app = Flask(__name__)

# -----------------------------
# Dropdown options (categorical)
# -----------------------------
DROPDOWN_OPTIONS = {
    "continent": ["Asia", "Europe", "North America", "South America", "Africa", "Oceania"],
    "education_of_employee": ["High School", "Bachelor's", "Master's", "Doctorate"],
    "has_job_experience": ["Y", "N"],
    "requires_job_training": ["Y", "N"],
    "region_of_employment": ["Northeast", "Midwest", "South", "West"],
    "unit_of_wage": ["Hour", "Week", "Month", "Year"],
    "full_time_position": ["Y", "N"]
}

# -----------------------------
# Home route → redirect to prediction form
# -----------------------------
@app.route('/')
def home():
    return render_template('result.html', dropdown_options=DROPDOWN_OPTIONS)

# -----------------------------
# Single prediction route
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        continent = request.form['continent']
        education_of_employee = request.form['education_of_employee']
        has_job_experience = request.form['has_job_experience']
        requires_job_training = request.form['requires_job_training']
        no_of_employees = int(request.form['no_of_employees'])
        yr_of_estab = int(request.form['yr_of_estab'])
        region_of_employment = request.form['region_of_employment']
        prevailing_wage = float(request.form['prevailing_wage'])
        unit_of_wage = request.form['unit_of_wage']
        full_time_position = request.form['full_time_position']

        # Calculate derived feature
        company_age = datetime.now().year - yr_of_estab

        # Prepare input
        data = USvisaData(
            continent, education_of_employee, has_job_experience, requires_job_training,
            no_of_employees, region_of_employment, prevailing_wage, unit_of_wage,
            full_time_position, company_age
        )

        input_df = data.get_usvisa_input_data_frame()
        predictor = USvisaClassifier()
        prediction = predictor.predict(input_df)
        pred_value = int(prediction[0])
        result_text = "Visa Approved" if pred_value == 1 else "Visa Denied"

        # Pass dropdown options and form values to retain selection
        form_values = request.form.to_dict()
        return render_template('result.html', 
                               dropdown_options=DROPDOWN_OPTIONS, 
                               result=result_text,
                               form_values=form_values)

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return render_template('result.html', dropdown_options=DROPDOWN_OPTIONS, result=f"Error: {e}")

# -----------------------------
# Batch prediction route
# -----------------------------
@app.route('/batch', methods=['GET', 'POST'])
def predict_batch():
    if request.method == 'POST':
        try:
            uploaded_file = request.files['file']
            if uploaded_file.filename == '':
                return render_template('batch.html', message="No file selected. Please upload a CSV file.")

            # Ensure artifacts folder exists
            output_dir = 'artifacts'
            os.makedirs(output_dir, exist_ok=True)

            file_path = os.path.join(output_dir, 'uploaded_batch.csv')
            uploaded_file.save(file_path)

            df = pd.read_csv(file_path)

            if 'yr_of_estab' in df.columns:
                df['company_age'] = datetime.now().year - df['yr_of_estab']
            else:
                return render_template('batch.html', message="CSV must contain 'yr_of_estab' column.")
            
            model = USvisaClassifier()
            predictions = model.predict(df)
            df['prediction'] = ['Certified' if p == 1 else 'Denied' for p in predictions]

            output_path = os.path.join(output_dir, 'batch_predictions.csv')
            df.to_csv(output_path, index=False)

            return render_template('batch.html',
                                   message="Batch prediction completed successfully!",
                                   download_link='batch_predictions.csv')

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            return render_template('batch.html', message=f"Error: {e}")

    return render_template('batch.html')

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Route to download the batch prediction results."""
    # Ensure the file is in the 'artifacts' directory for security
    file_path = os.path.join('artifacts', filename)
    return send_file(file_path, as_attachment=True)

# -----------------------------
# Run the app
# -----------------------------
if __name__ == "__main__":
    os.makedirs('artifacts', exist_ok=True)
    app.run(host='0.0.0.0', port=8080, debug=True)
