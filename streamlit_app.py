import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import os
import cv2

# ========== Page Configuration ==========
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== Custom CSS ==========
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary: #0077b6;
        --secondary: #00b4d8;
        --success: #2ecc71;
        --danger: #e74c3c;
        --warning: #f39c12;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #0077b6, #00b4d8);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 20px rgba(0, 119, 182, 0.2);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Info cards */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 119, 182, 0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #66b3ff;
    }
    
    .info-card h3 {
        color: #0077b6;
        margin-bottom: 0.8rem;
    }
    
    /* Result cards */
    .result-card {
        background: linear-gradient(135deg, #f8fcff, #e6f7ff);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #0077b6;
        box-shadow: 0 8px 25px rgba(0, 119, 182, 0.15);
        margin: 1rem 0;
    }
    
    .diagnosis-label {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2c3e50;
    }
    
    .diagnosis-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0077b6;
        margin: 0.5rem 0;
    }
    
    .confidence-bar {
        height: 30px;
        background: linear-gradient(90deg, #0077b6, #00b4d8);
        border-radius: 15px;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    
    /* Danger levels */
    .danger-high { border-left-color: #e74c3c; }
    .danger-medium { border-left-color: #f39c12; }
    .danger-low { border-left-color: #2ecc71; }
    .danger-none { border-left-color: #3498db; }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid #e9ecef;
        color: #6c757d;
    }
    
    .footer a {
        color: #0077b6;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.3s ease;
    }
    
    .footer a:hover {
        color: #00b4d8;
        text-decoration: underline;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #0077b6, #00b4d8);
        color: white;
        border-radius: 50px;
        padding: 0.5rem 2rem;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 119, 182, 0.4);
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #f8fcff;
    }
    
    /* Stats display */
    .stat-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0077b6;
    }
    
    .stat-label {
        color: #6c757d;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ========== Constants ==========
CLASS_LABELS = {
    0: 'Glioma',
    1: 'Meningioma',
    3: 'Pituitary',
    2: 'No Tumor'
}

DISEASE_INFO = {
    'Glioma': {
        'description': 'A tumor that occurs in the brain and spinal cord, originating from glial cells.',
        'location': 'Brain and spinal cord glial cells',
        'danger': 'High (can be malignant)',
        'danger_level': 'high',
        'advice': 'Consult a neuro-oncologist immediately. Treatment may include surgery, radiation, and chemotherapy.',
        'icon': '🧠',
        'color': '#e74c3c'
    },
    'Meningioma': {
        'description': 'A tumor that arises from the meninges, the membranes surrounding the brain and spinal cord.',
        'location': 'Meninges (brain and spinal cord membranes)',
        'danger': 'Medium (usually benign but can grow large)',
        'danger_level': 'medium',
        'advice': 'Consult a neurologist. Treatment often involves monitoring or surgery depending on size and symptoms.',
        'icon': '📊',
        'color': '#f39c12'
    },
    'Pituitary': {
        'description': 'A tumor that forms in the pituitary gland, affecting hormone production.',
        'location': 'Pituitary gland (base of the brain)',
        'danger': 'Low to Medium (typically benign)',
        'danger_level': 'low',
        'advice': 'Consult an endocrinologist. Treatment may include medication or surgery to restore hormone balance.',
        'icon': '🎯',
        'color': '#2ecc71'
    },
    'No Tumor': {
        'description': 'No signs of tumorous growth detected in the brain scan.',
        'location': 'N/A',
        'danger': 'None',
        'danger_level': 'none',
        'advice': 'Continue regular check-ups. Maintain a healthy lifestyle with proper diet and exercise.',
        'icon': '✅',
        'color': '#3498db'
    }
}

STATS_FILE = 'visitor_stats.json'

# ========== Helper Functions ==========
@st.cache_resource
def load_tumor_model():
    """Load the trained model with caching"""
    try:
        model = load_model('models/final_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(img, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def get_feature_maps(model, img_array, layer_indices=[2, 5, 8]):
    """Extract feature maps from specified layers"""
    feature_maps = []
    layer_outputs = [model.layers[i].output for i in layer_indices if i < len(model.layers)]
    
    if layer_outputs:
        feature_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
        features = feature_model.predict(img_array)
        
        for i, feature in enumerate(features):
            feature_maps.append({
                'layer_index': layer_indices[i],
                'layer_name': model.layers[layer_indices[i]].name,
                'feature': feature
            })
    
    return feature_maps

def update_visitor_stats():
    """Update visitor statistics"""
    if not os.path.exists(STATS_FILE):
        stats = {
            "total_visits": 0,
            "unique_visitors": 0,
            "streamlit_visits": 0
        }
    else:
        try:
            with open(STATS_FILE, 'r') as f:
                stats = json.load(f)
        except:
            stats = {
                "total_visits": 0,
                "unique_visitors": 0,
                "streamlit_visits": 0
            }
    
    # Update stats
    if 'streamlit_visits' not in stats:
        stats['streamlit_visits'] = 0
    
    stats['streamlit_visits'] += 1
    stats['total_visits'] = stats.get('total_visits', 0) + 1
    
    # Save updated stats
    try:
        with open(STATS_FILE, 'w') as f:
            json.dump(stats, f, indent=4)
    except:
        pass
    
    return stats

def create_confidence_chart(predictions, class_labels):
    """Create a bar chart for prediction confidence"""
    classes = [class_labels[i] for i in sorted(class_labels.keys())]
    confidences = [predictions[0][i] * 100 for i in sorted(class_labels.keys())]
    
    fig = go.Figure(data=[
        go.Bar(
            x=confidences,
            y=classes,
            orientation='h',
            marker=dict(
                color=confidences,
                colorscale='Blues',
                showscale=False
            ),
            text=[f'{c:.2f}%' for c in confidences],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence for All Classes",
        xaxis_title="Confidence (%)",
        yaxis_title="Tumor Type",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

# ========== Main Application ==========
def main():
    # Update visitor stats
    if 'visited' not in st.session_state:
        stats = update_visitor_stats()
        st.session_state.visited = True
        st.session_state.stats = stats
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Brain Tumor Classification System</h1>
        <p>Upload an MRI scan to detect and classify brain tumors using advanced AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/brain.png", width=100)
        st.title("Navigation")
        
        page = st.radio(
            "Select Page",
            ["🏠 Home - Upload & Analyze", "📚 Tumor Information", "📊 Statistics"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Quick stats
        if 'stats' in st.session_state:
            st.markdown("### Quick Stats")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Visits", st.session_state.stats.get('total_visits', 0))
            with col2:
                st.metric("Streamlit Visits", st.session_state.stats.get('streamlit_visits', 0))
        
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        This AI-powered system analyzes brain MRI scans to detect and classify tumors into four categories:
        - Glioma
        - Meningioma
        - Pituitary Tumor
        - No Tumor
        """)
    
    # Main content based on selected page
    if page == "🏠 Home - Upload & Analyze":
        show_upload_page()
    elif page == "📚 Tumor Information":
        show_tumor_info_page()
    elif page == "📊 Statistics":
        show_statistics_page()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p><strong>Brain Tumor Classification System</strong> © 2024</p>
        <p>Developed by <a href="https://www.linkedin.com/in/mo-abdalkader/" target="_blank">Mohamed Abdalkader</a></p>
        <p>
            <a href="mailto:Mohameed.Abdalkadeer@gmail.com">📧 Email</a> | 
            <a href="https://github.com/Mo-Abdalkader" target="_blank">💻 GitHub</a>
        </p>
        <p style="font-size: 0.9rem; color: #999; margin-top: 1rem;">
            ⚠️ Disclaimer: This application is for educational purposes only. 
            Always consult healthcare professionals for medical diagnosis.
        </p>
    </div>
    """, unsafe_allow_html=True)

def show_upload_page():
    """Main upload and analysis page"""
    # Load model
    model = load_tumor_model()
    
    if model is None:
        st.error("❌ Model could not be loaded. Please check the model file.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # File uploader
    st.markdown("### 📤 Upload MRI Scan")
    uploaded_file = st.file_uploader(
        "Choose an MRI image file (JPEG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a brain MRI scan for tumor detection"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📸 Uploaded Image")
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_column_width=True)
        
        with col2:
            st.markdown("#### 🔬 Analysis Options")
            
            show_feature_maps = st.checkbox("Show Feature Maps", value=False)
            show_all_predictions = st.checkbox("Show All Class Predictions", value=True)
            
            analyze_button = st.button("🔍 Analyze Image", type="primary", use_container_width=True)
        
        if analyze_button:
            with st.spinner("🔄 Analyzing image..."):
                # Preprocess and predict
                processed_img = preprocess_image(image)
                predictions = model.predict(processed_img)
                predicted_class = np.argmax(predictions[0])
                confidence = float(np.max(predictions[0]))
                class_name = CLASS_LABELS.get(predicted_class, 'Unknown')
                
                # Display results
                st.markdown("---")
                st.markdown("## 📋 Analysis Results")
                
                # Result card
                info = DISEASE_INFO[class_name]
                st.markdown(f"""
                <div class="result-card">
                    <div style="text-align: center;">
                        <span style="font-size: 3rem;">{info['icon']}</span>
                        <p class="diagnosis-label">Diagnosis</p>
                        <p class="diagnosis-value">{class_name}</p>
                        <p class="diagnosis-label">Confidence: {confidence * 100:.2f}%</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence visualization
                st.markdown("#### Confidence Level")
                st.progress(confidence)
                
                # Detailed information
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="info-card danger-{info['danger_level']}">
                        <h3>📝 Description</h3>
                        <p>{info['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="info-card danger-{info['danger_level']}">
                        <h3>📍 Location</h3>
                        <p>{info['location']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="info-card danger-{info['danger_level']}">
                        <h3>⚠️ Danger Level</h3>
                        <p>{info['danger']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="info-card danger-{info['danger_level']}">
                        <h3>💡 Recommended Actions</h3>
                        <p>{info['advice']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # All class predictions
                if show_all_predictions:
                    st.markdown("---")
                    st.markdown("### 📊 Confidence for All Classes")
                    fig = create_confidence_chart(predictions, CLASS_LABELS)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feature maps
                if show_feature_maps:
                    st.markdown("---")
                    st.markdown("### 🗺️ Feature Maps Visualization")
                    st.info("Feature maps show what the neural network 'sees' at different layers")
                    
                    with st.spinner("Generating feature maps..."):
                        feature_maps = get_feature_maps(model, processed_img)
                        
                        if feature_maps:
                            for fm in feature_maps:
                                st.markdown(f"#### Layer {fm['layer_index']}: {fm['layer_name']}")
                                
                                # Display first 16 feature maps
                                features = fm['feature'][0]
                                n_features = min(16, features.shape[-1])
                                
                                cols = st.columns(4)
                                for i in range(n_features):
                                    with cols[i % 4]:
                                        feature_img = features[:, :, i]
                                        feature_img = (feature_img - feature_img.min()) / (feature_img.max() - feature_img.min() + 1e-8)
                                        st.image(feature_img, caption=f"Filter {i+1}", use_container_width=True)
                        else:
                            st.warning("Could not generate feature maps")

def show_tumor_info_page():
    """Page showing detailed information about tumor types"""
    st.markdown("## 📚 Brain Tumor Types - Detailed Information")
    
    # Tumor type selector
    selected_tumor = st.selectbox(
        "Select a tumor type to learn more:",
        list(DISEASE_INFO.keys())
    )
    
    info = DISEASE_INFO[selected_tumor]
    
    # Display detailed information
    st.markdown(f"""
    <div class="result-card">
        <div style="text-align: center;">
            <span style="font-size: 4rem;">{info['icon']}</span>
            <h1>{selected_tumor}</h1>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="info-card danger-{info['danger_level']}">
            <h3>📝 Description</h3>
            <p>{info['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-card danger-{info['danger_level']}">
            <h3>📍 Location</h3>
            <p>{info['location']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="info-card danger-{info['danger_level']}">
            <h3>⚠️ Danger Level</h3>
            <p>{info['danger']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-card danger-{info['danger_level']}">
            <h3>💡 Recommended Actions</h3>
            <p>{info['advice']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional information based on tumor type
    if selected_tumor == "Glioma":
        with st.expander("🔬 Treatment Options"):
            st.markdown("""
            - **Surgery**: Remove as much tumor as safely possible
            - **Radiation Therapy**: Kill remaining tumor cells
            - **Chemotherapy**: Often using temozolomide (Temodar)
            - **Targeted Therapy**: For tumors with specific genetic mutations
            - **Tumor Treating Fields**: Newer approach using electrical fields
            """)
    elif selected_tumor == "Meningioma":
        with st.expander("🔬 Treatment Options"):
            st.markdown("""
            - **Observation**: For small, asymptomatic meningiomas
            - **Surgery**: Complete removal when possible
            - **Radiation Therapy**: For tumors that cannot be fully removed
            - **Stereotactic Radiosurgery**: Precisely targeted radiation
            """)
    elif selected_tumor == "Pituitary":
        with st.expander("🔬 Treatment Options"):
            st.markdown("""
            - **Medication**: To shrink tumor or control hormone production
            - **Transsphenoidal Surgery**: Removal through nasal passage
            - **Radiation Therapy**: For tumors that cannot be fully removed
            - **Hormone Replacement Therapy**: If normal hormone production is affected
            """)
    else:  # No Tumor
        with st.expander("🧘 Brain Health Tips"):
            st.markdown("""
            - **Mental Exercise**: Keep your mind active with puzzles and learning
            - **Physical Activity**: Regular exercise improves blood flow to the brain
            - **Healthy Diet**: Foods rich in antioxidants and omega-3 fatty acids
            - **Quality Sleep**: Aim for 7-9 hours of restorative sleep
            - **Stress Management**: Practice relaxation techniques
            """)

def show_statistics_page():
    """Page showing application statistics"""
    st.markdown("## 📊 Application Statistics")
    
    # Load stats
    try:
        with open(STATS_FILE, 'r') as f:
            stats = json.load(f)
    except:
        stats = {
            "total_visits": 0,
            "unique_visitors": 0,
            "streamlit_visits": 0,
            "ip_data": {}
        }
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-value">{}</div>
            <div class="stat-label">Total Visits</div>
        </div>
        """.format(stats.get('total_visits', 0)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-value">{}</div>
            <div class="stat-label">Unique Visitors</div>
        </div>
        """.format(stats.get('unique_visitors', 0)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-value">{}</div>
            <div class="stat-label">Streamlit App Visits</div>
        </div>
        """.format(stats.get('streamlit_visits', 0)), unsafe_allow_html=True)
    
    # IP data visualization
    if stats.get('ip_data'):
        st.markdown("---")
        st.markdown("### 🌐 Visits by IP Address")
        
        ip_data = stats['ip_data']
        ips = list(ip_data.keys())[:10]  # Top 10
        visits = [ip_data[ip] for ip in ips]
        
        fig = go.Figure(data=[
            go.Bar(
                x=ips,
                y=visits,
                marker=dict(
                    color=visits,
                    colorscale='Blues',
                    showscale=False
                ),
                text=visits,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Top 10 IP Addresses by Visits",
            xaxis_title="IP Address",
            yaxis_title="Number of Visits",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Refresh button
    if st.button("🔄 Refresh Statistics", use_container_width=True):
        st.rerun()

# Run the main application
main()
