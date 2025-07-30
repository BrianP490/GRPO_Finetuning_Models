# My First Streamlit App

This is a simple "Hello World" Streamlit application that demonstrates basic interactive widgets.

## Setup

1.  **Clone the repository (if applicable):**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
    (Skip this step if you're just creating the files locally.)

2.  **Create a virtual environment (recommended):**
    ```bash
    conda env create -f ./streamlit/streamlit_env.yaml
    ```

3.  **Activate the virtual environment:**
    ```conda activate strl_env```

4.  **Install dependencies:**
    ```bash
    pip install -r ./streamlit/requirements.txt
    ```

## How to Run

Make sure your virtual environment is activated, navigate to the streamlit directory, then run the Streamlit app from your terminal:

```bash
cd streamlit
streamlit run ./app/main.py