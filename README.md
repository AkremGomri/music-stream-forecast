# Music Stream Forecasting App

## 🌐 Live Demo
Try the deployed version of the app here: **[https://music-stream-forecast-ukrwopgzazcoyaruqcsrer.streamlit.app/](https://music-stream-forecast-ukrwopgzazcoyaruqcsrer.streamlit.app/)**

---

## 📌 Project Context

The company **Dibsteur** launched an artificial intelligence development seminar with the goal of building an AI-based solution capable of detecting **promising music artists**. The company wants to sign artists who:

* Will generate **enough future streams** to be profitable,
* But **not too many**, so they do not attract major labels and investors.

This project provides:

* A forecasting model to predict an artist's future performance,
* A computation of **confidence intervals**, enabling a better understanding of prediction uncertainty,
* A **Streamlit web app** that displays predictions, uncertainty ranges, and cleaned data.

A Colab notebook containing the model parameters and data preparation steps is available here:
[https://colab.research.google.com/drive/1XG_FTA8m5sVqWvcB0jMfyjl1KWLXSPDM?create=true](https://colab.research.google.com/drive/1XG_FTA8m5sVqWvcB0jMfyjl1KWLXSPDM?create=true)

---

## 🚀 How to Run Locally

### 1. Clone the project

```bash
git clone https://github.com/yourusername/music-stream-forecast.git
cd music-stream-forecast
```

### 2. Install dependencies

Ensure you have Python 3.9+ installed.

```bash
pip install -r requirements.txt
```

### 3. Add your Nixtla API key

Create an account on: [https://www.nixtla.io/](https://www.nixtla.io/)

Generate your API token, then create the following file:

```
.streamlit/secrets.toml
```

Add inside:

```toml
NIXTLA_API_KEY = "YOUR_API_KEY" # <-- replace YOUR_API_KEY with your actual API key
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 🐳 Running with Docker

You can run the app directly from Docker Hub using the published image:

**Image:** `akremgomri1/music-stream-forecast:latest`

### 1️⃣ Create your Nixtla secrets file

Create the file:

```
.streamlit/secrets.toml
```

Add your API key:

```toml
NIXTLA_API_KEY = "YOUR_API_KEY" # <-- replace YOUR_API_KEY with your actual API key
```

---

### 2️⃣ Run the app using Docker Hub image

#### **Windows Git Bash (recommended)**

```bash
docker run --name music-forecast -p 8501:8501 -v "$(pwd -W)/.streamlit:/app/.streamlit" akremgomri1/music-stream-forecast:latest
```

#### **Windows PowerShell**

```powershell
docker run -p 8501:8501 ` -v ${PWD}\.streamlit:/app/.streamlit ` akremgomri1/music-stream-forecast:latest
```

#### **Windows CMD**

```cmd
docker run -p 8501:8501 -v %cd%\.streamlit:/app/.streamlit akremgomri1/music-stream-forecast:latest
```

#### **Linux & macOS**

```bash
docker run -p 8501:8501 -v $(pwd)/.streamlit:/app/.streamlit akremgomri1/music-stream-forecast:latest
```

The app will be available at:
👉 **[http://localhost:8501](http://localhost:8501)**

---

## 📁 Project Structure

```
├── app.py
├── requirements.txt
├── Dockerfile
├── .streamlit/
│   └── secrets.toml (ignored by git)
├── data/
└── models/
```

---

## 🛠️ Technologies Used

* **Python**
* **Streamlit** (frontend interface)
* **Nixtla TimeGPT API** (time‑series forecasting)
* **Pandas / NumPy** (data processing)
* **Docker** (containerization)

---

## 📄 License

MIT License.

---

## 👤 Author & Creator

**Akrem GOMRI** — AI Engineer & Data Scientist

Creator and sole developer of the Music Stream Forecasting App  

🌐 Website: [https://akremgomri.github.io/](https://akremgomri.github.io/)  
<img src="https://img.icons8.com/ios-glyphs/16/000000/linkedin.png"/> LinkedIn: [https://www.linkedin.com/in/akrem-gomri/](https://www.linkedin.com/in/akrem-gomri/)

---
