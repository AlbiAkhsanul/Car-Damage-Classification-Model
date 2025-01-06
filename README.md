# 🚗 **Car Damage Detection and Classification**  

Welcome to the  repository for Riset Informatika Repository. This project features two core machine learning models tailored to detect car damage and clasify car damage.  

---

## 🤖 **AI-Powered Model for Vehicle Damage Detection and Classification**

Our application integrates two specialized models, each designed to handle a specific aspect of car damage analysis:  

### 🔍 **Skin Type Model**  
- **Architecture**: YOLOv8n  
- **Why YOLOv8n?**  
   is lightweight, fast, and highly efficient for object detection tasks, making it ideal for detecting damaged regions on vehicles.  
- **Goal**: EAccurately detect areas of damage on vehicle images and draw bounding boxes around the detected regions.  

### 🗂️ **Skin Conditions Model**  
- **Architecture**: MobileNetV2 (Transfer Learning)  
- **Why MobileNetV2?**  
   MobileNetV2 is optimized for recognizing patterns and features, making it an excellent choice for classifying damage severity. Additionally, its lightweight design ensures efficiency.  
- **Goal**: Classify the severity of damage into three categories: Minor, Moderate, Severe.  

---

## 🛠️ **Main Features**  

- 🔍 Damage Detection: Precisely identifies damaged areas on a vehicle.
- 🗂️ Damage Classification: Classifies the severity of each detected damage.
- 📊 Summary Report: Provides a detailed summary of damages detected in different regions.

---

## 🧠 **How To Use**

- 📤 Upload Image: Start by uploading a vehicle image through the application interface.
- 🖼️ View Bounding Boxes: See the detected damages highlighted with bounding boxes on the uploaded image.
- 📋 Check Classification: Review the classification of damage severity for each detected region.
- ✅ Summary: Get a comprehensive summary of all detected damages, categorized by region.

---
## 🙌 Acknowledgments
Special thanks to:

- The open datasets like COCO for providing high-quality data used during training.
- The community behind YOLOv8 and MobileNetV2 for their contributions to accessible machine learning architectures.
---

We’re excited to bring you a cutting-edge solution for car damage detection and classification, powered by AI. Your feedback and contributions are welcome to make this project even better!