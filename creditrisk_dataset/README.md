# Credit Risk Prediction Application

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A machine learning-powered web application for predicting credit risk using the German Credit Dataset. The application uses an Extra Trees Classifier to assess whether a credit applicant poses a good or bad risk based on various demographic and financial features.

## Table of Contents

- [What the Project Does](#what-the-project-does)
- [Why the Project is Useful](#why-the-project-is-useful)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Deployment](#deployment)
  - [Streamlit Cloud](#streamlit-cloud-recommended)
  - [Heroku](#heroku)
  - [Docker](#docker)
  - [Other Platforms](#other-platforms)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Support](#support)
- [License](#license)

## What the Project Does

This project provides an interactive web application that predicts credit risk for loan applicants. Users can input applicant information including:

- **Demographics**: Age, Sex, Job type
- **Housing**: Housing status (own, rent, free)
- **Financial**: Saving accounts, Checking account status
- **Loan Details**: Credit amount, Duration (months), Purpose

The application then uses a pre-trained Extra Trees Classifier model to predict whether the credit risk is:
- **Good (1)**: Lower risk applicant
- **Bad (0)**: Higher risk applicant

## Why the Project is Useful

- **Automated Risk Assessment**: Quickly evaluate credit applications without manual analysis
- **Consistent Decision Making**: ML model provides consistent predictions based on data patterns
- **User-Friendly Interface**: Simple Streamlit web interface accessible to non-technical users
- **Production-Ready**: Includes deployment configurations for multiple platforms
- **Educational Value**: Demonstrates end-to-end ML workflow from data analysis to deployment

## Key Features

- 🎯 **Real-time Predictions**: Instant credit risk assessment
- 📊 **Interactive Web Interface**: Easy-to-use Streamlit dashboard
- 🤖 **Machine Learning Model**: Pre-trained Extra Trees Classifier
- 🔄 **Label Encoding**: Automatic handling of categorical features
- 🚀 **Multiple Deployment Options**: Streamlit Cloud, Heroku, Docker support
- 📦 **Pre-trained Models**: Ready-to-use model and encoders included

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd creditrisk_dataset
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Or using a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   Ensure all required files are present:
   - `app.py`
   - `extra_trees_model.pkl`
   - `Sex_encoder.pkl`
   - `Housing_encoder.pkl`
   - `Saving accounts_encoder.pkl`
   - `Checking account_encoder.pkl`
   - `Purpose_encoder.pkl`

### Usage

1. **Run the application:**
   ```bash
   streamlit run app.py
   ```

2. **Access the app:**
   - The app will automatically open in your default browser
   - Default URL: `http://localhost:8501`

3. **Make predictions:**
   - Fill in the applicant information using the form fields
   - Click the "Predict Risk" button
   - View the prediction result (Good or Bad credit risk)

## Project Structure

```
creditrisk_dataset/
├── app.py                          # Main Streamlit application
├── analysis_model.ipynb            # Jupyter notebook with EDA and model training
├── extra_trees_model.pkl           # Trained Extra Trees Classifier model
├── Sex_encoder.pkl                 # Label encoder for Sex feature
├── Housing_encoder.pkl             # Label encoder for Housing feature
├── Saving accounts_encoder.pkl     # Label encoder for Saving accounts feature
├── Checking account_encoder.pkl    # Label encoder for Checking account feature
├── Purpose_encoder.pkl             # Label encoder for Purpose feature
├── german_credit_data.csv          # German Credit Dataset
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker configuration
├── Procfile                        # Heroku deployment configuration
├── setup.sh                        # Heroku setup script
└── README.md                       # Project documentation
```

## Model Details

- **Algorithm**: Extra Trees Classifier (Extremely Randomized Trees)
- **Dataset**: German Credit Dataset (522 samples after preprocessing)
- **Features**: 8 input features (Age, Sex, Job, Housing, Saving accounts, Checking account, Credit amount, Duration)
- **Target**: Binary classification (Good/Bad credit risk)
- **Preprocessing**: Label encoding for categorical variables, missing value handling

The model was trained and evaluated in `analysis_model.ipynb`. For detailed analysis and model development, refer to the notebook.

## Deployment

### Streamlit Cloud (Recommended)

**Free and easy deployment option:**

1. **Push code to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository and branch
   - Set main file path to `app.py`
   - Click "Deploy"

3. **Your app will be live at:** `https://your-app-name.streamlit.app`

**Note:** Ensure all `.pkl` files are committed to your repository.

### Heroku

1. **Install Heroku CLI** and login:
   ```bash
   heroku login
   ```

2. **Create Heroku app:**
   ```bash
   heroku create your-app-name
   ```

3. **Deploy:**
   ```bash
   git push heroku main
   ```

The `Procfile` and `setup.sh` are already configured for Heroku deployment.

### Docker

1. **Build the Docker image:**
   ```bash
   docker build -t credit-risk-app .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8501:8501 credit-risk-app
   ```

3. **Access the app:** `http://localhost:8501`

### Other Platforms

The application can be deployed on:
- **AWS EC2**: Set up EC2 instance, install dependencies, run Streamlit app
- **Google Cloud Platform**: Use Cloud Run or Compute Engine
- **Azure**: Deploy using Azure App Service or Container Instances
- **DigitalOcean**: Use App Platform or Droplets

For production deployments, consider:
- Adding authentication/authorization
- Using a reverse proxy (nginx)
- Setting up SSL/TLS certificates
- Monitoring and logging

## Troubleshooting

### Common Issues

**Model file not found:**
- Ensure all `.pkl` files are in the same directory as `app.py`
- Check file paths are correct

**Port already in use:**
```bash
streamlit run app.py --server.port=8502
```

**Import errors:**
- Verify all packages in `requirements.txt` are installed
- Check Python version (3.8+ required)
- Try reinstalling dependencies: `pip install -r requirements.txt --upgrade`

**ValueError during prediction:**
- Ensure input values match expected formats
- Check that categorical values match training data categories

**Missing dependencies:**
- Run `pip install -r requirements.txt` again
- Ensure virtual environment is activated if using one

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Contribution Guidelines

- Follow PEP 8 style guidelines for Python code
- Add comments and docstrings for new functions
- Update documentation as needed
- Test your changes before submitting

## Support

- **Issues**: Report bugs or request features via [GitHub Issues](https://github.com/your-username/credit-risk-project/issues)
- **Documentation**: Check `analysis_model.ipynb` for detailed model development process
- **Questions**: Open a discussion in the repository's Discussions section

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This application is for educational and demonstration purposes. For production use in financial services, ensure compliance with relevant regulations and perform thorough model validation.
