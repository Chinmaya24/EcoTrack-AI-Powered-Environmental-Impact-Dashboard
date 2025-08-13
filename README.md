# EcoTrack – AI-Powered Environmental Impact Dashboard

> **EcoTrack** is a web-based AI dashboard for monitoring and visualizing environmental and energy data. Using Prophet forecasting, interactive charts, and automated email reports, it helps track trends, reduce ecological impact, and support sustainability projects, research, and decision-making.

---

## 📖 Overview
EcoTrack is a web-based application designed to help individuals, organizations, and researchers monitor, analyze, and visualize environmental and energy consumption data. It integrates AI-powered forecasting (via Prophet), automated PDF/email reporting, and interactive dashboards to turn raw environmental data into actionable insights.

---

## ✨ Features
- 📊 **Interactive Dashboard** – Visualize historical and real-time data.
- 🔮 **AI Forecasting** – Predict future trends using Prophet.
- 📂 **Data Management** – Store and retrieve datasets with PostgreSQL.
- 📧 **Automated Reports** – Generate and email PDF summaries via Google API.
- 🖥 **Web Interface** – Built with Flask for easy access and scalability.
- 📈 **Advanced Analytics** – Powered by Pandas, SciPy, and Statsmodels.

---

## 🛠 Tech Stack
- **Backend:** Flask, Flask-SQLAlchemy
- **Database:** PostgreSQL
- **Forecasting:** Prophet, Statsmodels, SciPy
- **Visualization & Reporting:** Pandas, ReportLab, xhtml2pdf
- **Automation:** Google API, python-dotenv
- **System Utilities:** GPUtil, psutil

---

## 📦 Installation

1. **Clone the Repository**
```bash
git clone https://github.com/your-username/ecotrack.git
cd ecotrack
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure Environment Variables**
Create a `.env` file inside the `main` directory:
```
DATABASE_URL=postgresql://username:password@localhost:5432/ecotrack
GOOGLE_CLIENT_SECRET=your_google_client_secret
```

---

## ▶️ Running the Application
```bash
cd "hackathon final/main"
python app.py
```
The app will run locally on **http://127.0.0.1:5000**.

---

## 📧 Running Email Reporting Service
```bash
cd "hackathon final/main/email"
python app.py
```
Make sure `client_secrets.json` is configured for Google API access.

---

## 📊 How the Model Works
- Data is uploaded or fetched from the database.
- Preprocessed using Pandas and SciPy.
- Prophet is used for trend forecasting.
- Results are displayed on the dashboard and included in PDF reports.

---

## 📜 License
This project is licensed under the MIT License.
