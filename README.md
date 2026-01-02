# Brain Tumor Classification System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.0-lightgrey)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.6-orange)

A web application to detect and classify brain tumors from MRI scans using deep learning.

![Brain Tumor Classification](https://www.mhsi.com/blog/wp-content/uploads/2021/06/BrainTumor-1250205787-1200x772.jpg)

---

## 🎯 Live Demo

### 🌐 **Try It Now!**

<div align="center">

[![Streamlit App](https://img.shields.io/badge/🧠_Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://brain-tumor-cls.streamlit.app//)

**[🚀 Launch Interactive Demo](https://brain-tumor-cls.streamlit.app/)**

*Experience the full-featured AI-powered brain tumor classifier with real-time predictions and feature maps visualization!*

</div>

---

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

## Usage

1. Upload an MRI scan through the interface:
   - Drag and drop an image file
   - Or click to browse and select a file

2. View the analysis results:
   - Tumor classification
   - Confidence level
   - Detailed information about the detected condition
   - Recommendations for next steps

## Development

### Project Structure

```
brain-tumor-classification/
├── models/                 # Trained machine learning models
│   └── final_model.keras
├── visitor_stats.json      # Analytics data
└── requirements.txt        # Python dependencies
```

### Model Details

The deep learning model uses a convolutional neural network architecture trained on a dataset of MRI scans. It achieves classification by identifying patterns in brain tissue that correspond to different tumor types.

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



