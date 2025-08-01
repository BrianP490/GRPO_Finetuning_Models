---
title: Spam Classifier Agent
emoji: 🚀
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: Streamlit Demo App
license: apache-2.0
---

# 📧 Spam Classifier Agent Live Demo

A Streamlit web application that leverages a fine-tuned GPT-2 model to classify text messages as either "SPAM" or "NOT SPAM". This agent helps users quickly determine the nature of a message. Try it here @ [SoggyBurritos/Spam_Classifier_Agent](https://huggingface.co/spaces/SoggyBurritos/Spam_Classifier_Agent)

## ✨ Features

* **Text Input:** Easily enter any text message for classification.
* **Real-time Prediction:** Get instant results on whether the message is spam or not.
* **Probability Scores:** View the confidence scores for both "SPAM" and "NOT SPAM" categories.
* **Intuitive UI:** A clean and user-friendly interface built with Streamlit.
* **Fine-tuned GPT-2 Model:** Utilizes a powerful transformer model for accurate classification.

## 💻 How to Use

1.  **Enter Text:** Type or paste the message you want to classify into the provided text area.
2.  **Analyze:** Click the "Analyze Text" button.
3.  **View Results:** The application will display the prediction (SPAM or NOT SPAM) along with the probability scores.

## ⚙️ Technical Details for Hugging Face Spaces

**App File Location:**
The main Streamlit application script is located at `streamlit/app/main.py`. Hugging Face Spaces will be configured to run this specific file.

**Dependencies:**
All required Python packages for this deployment is listed in `requirements.txt`.

**Model & Data:**
* The fine-tuned GPT-2 model weights (`Spam-Classifier-GPT2-Model.pt`) are expected to be in the `models/` directory at the root of the repository.
* The application also relies on data preparation scripts which may download or create necessary data files.