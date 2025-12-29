# Brain Tumor Classification System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.0-lightgrey)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.6-orange)

A web application to detect and classify brain tumors from MRI scans using deep learning.

![Brain Tumor Classification](https://www.mhsi.com/blog/wp-content/uploads/2021/06/BrainTumor-1250205787-1200x772.jpg)

## Features

- **MRI Scan Analysis**: Upload brain MRI scans to detect presence of tumors
- **Multi-Class Classification**: Identifies four categories:
  - Glioma
  - Meningioma
  - Pituitary Tumor
  - No Tumor
- **Detailed Information**: Provides comprehensive details about each tumor type
- **Interactive UI**: Drag-and-drop interface with real-time feedback
- **Educational Content**: Learn about different brain tumor types and characteristics
- **Visitor Statistics**: Tracks unique visitors and total visits for analytics

## Technology Stack

- **Backend**: Python, Flask
- **Machine Learning**: TensorFlow, Keras
- **Frontend**: HTML, CSS, JavaScript
- **Authentication**: Flask-HTTPAuth
- **Image Processing**: PIL, NumPy
- **Analytics**: Custom visitor tracking

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Mo-Abdalkader/Projcts.git
   cd Projects\AI\02 Deep Learning\Computer Vision\Brain Tumor
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the pre-trained model:
   - Create a `models` directory in the project root
   - Place the trained model file (`final_model.keras`) in the `models` directory

5. Create required directories:
   ```bash
   mkdir -p static/uploads
   ```

## Usage

1. Start the application:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Upload an MRI scan through the interface:
   - Drag and drop an image file
   - Or click to browse and select a file

4. View the analysis results:
   - Tumor classification
   - Confidence level
   - Detailed information about the detected condition
   - Recommendations for next steps

## Admin Dashboard

Access visitor statistics through the admin dashboard:

1. Navigate to `/stats`
2. Login with admin credentials
3. View metrics including:
   - Total visits
   - Unique visitors
   - Current online users
   - IP-based statistics

## Development

### Project Structure

```
brain-tumor-classification/
├── app.py                  # Main Flask application
├── models/                 # Trained machine learning models
│   └── final_model.keras
├── static/                 # Static files (CSS, JS, images)
│   ├── css/
│   │   └── style.css
│   └── uploads/            # Temporary storage for uploads
├── templates/              # HTML templates
│   ├── index.html          # Main interface
│   └── stats.html          # Admin statistics dashboard
├── visitor_stats.json      # Analytics data
└── requirements.txt        # Python dependencies
```

### Model Details

The deep learning model uses a convolutional neural network architecture trained on a dataset of MRI scans. It achieves classification by identifying patterns in brain tissue that correspond to different tumor types.

## Security Features

- Password-protected admin dashboard
- Secure file uploads with validation
- Temporary file storage with automatic cleanup
- Session management for visitor tracking

## License

[MIT License](LICENSE)

## Contact

- **Developer**: Mohamed Abdalkader
- **Email**: Mohameed.Abdalkadeer@gmail.com
- **LinkedIn**: [mo-abdalkader](https://www.linkedin.com/in/mo-abdalkader/)
- **GitHub**: [Mo-Abdalkader](https://github.com/Mo-Abdalkader)

## Acknowledgments

- Thanks to all contributors who have helped with this project
- Special thanks to the open-source community for providing tools and libraries
- MRI dataset [source information]

---

> **Disclaimer**: This application is for educational and research purposes only. It should not be used for medical diagnosis. Always consult with healthcare professionals for medical advice.
