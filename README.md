🧠 AI-Based Multi-Input Recognition System

📘 Project Overview

The AI-Based Multi-Input Recognition System is an advanced multimodal application designed to process and understand inputs from multiple sources — speech, text, hand gestures, and images. It integrates Natural Language Processing (NLP), Computer Vision (CV), and Machine Learning (ML) techniques to enable seamless human–computer interaction through natural communication modes. This system provides an intelligent way to recognize, analyze, and respond to user inputs efficiently, bridging the gap between human and AI understanding.

⚙️ Key Features

Hand Gesture Recognition:
Uses the Random Forest Classifier to identify static hand gestures captured through a webcam. MediaPipe and OpenCV are used for landmark detection and feature extraction.

Optical Character Recognition (OCR):
Employs PyTesseract to extract and digitize text from images, making document reading and automation easier.

Speech Recognition:
Implements Google ASR (Automatic Speech Recognition) to convert spoken input into text with high accuracy and multilingual support.

Language Translation:
Utilizes Googletrans to translate recognized text into multiple languages for multilingual communication.

Sentiment Analysis:
Integrates BERT/TextBlob models to analyze the emotional tone of text, identifying positive, negative, or neutral sentiments.

Image Captioning:
Uses a VisionEncoderDecoderModel (ViT + GPT-2) for generating descriptive captions of uploaded images, enabling deeper visual understanding.

User Authentication and Data Logging:
Managed with SQLite3, providing secure login, user activity logs, and database integration for personalized tracking.

Email Integration:
Automatically sends processed outputs and results to users via email for record-keeping and accessibility.

🧩 Technology Stack

Frontend/UI: Streamlit

Backend: Python

Libraries: OpenCV, MediaPipe, PyTesseract, SpeechRecognition, Transformers, Torch, Pillow, Googletrans, TextBlob

Database: SQLite3

🚀 Objective

This project demonstrates how AI can combine multimodal inputs—voice, gesture, text, and image—to create a unified, intelligent communication system. It promotes accessibility, automation, and efficiency in real-world applications such as education, assistive technology, and interactive systems.
