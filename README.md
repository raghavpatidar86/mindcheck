# 🧠 MindCheck — AI-Powered Stress Prediction Web App

> **Understand. Manage. Thrive.**  
> A full-stack mental wellness platform that uses machine learning to assess stress levels, track your wellness journey over time, and deliver AI-powered personalized advice.

---

## 📸 Pages

| Page | Route | Description |
|------|-------|-------------|
| Home / Assessment | `/` | Stress assessment form + live results |
| Insights Dashboard | `/insights` | Trend analysis, charts, AI advice |
| User Profile | `/profile` | Profile management + history timeline |
| Login | `/login-page` | Secure authentication |
| Register | `/register` | New account creation |

---

## ✨ Features

- **ML Stress Prediction** — 7-question assessment scored by a trained classifier (Random Forest / Gradient Boosting / SVM / KNN / Logistic Regression — best model auto-selected)
- **Stress Score Breakdown** — Per-category scores across Emotional, Sleep, Overwhelm, Mood, Physical, Concentration, and Irritability
- **Animated Gauge Chart** — Visual stress level indicator with color-coded results
- **Wellness Insights Dashboard** — Trend detection (improving / stable / worsening), stress history chart, top stressors, personalized recommendations
- **AI Wellness Advisor** — Ollama-powered LLM generates tailored advice based on your latest assessment
- **User Profile Page** — Edit name, email, age, occupation, avatar; view assessment history timeline
- **Dark / Light Theme** — Persisted across all pages via localStorage
- **Secure Auth** — bcrypt password hashing, Flask session management
- **Responsive Design** — Works on desktop, tablet, and mobile

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Backend | Python 3, Flask |
| Database | MySQL |
| ML | scikit-learn (Random Forest, Gradient Boosting, SVM, KNN, Logistic Regression) |
| AI Advisor | Ollama (llama3.2) |
| 3D Background | Three.js |
| Animations | GSAP 3 + ScrollTrigger |
| Charts | Canvas API (custom) |
| Fonts | Google Fonts — Nunito + DM Sans |

---

## 📁 Project Structure

```
stress/
├── app.py                  # Flask backend — routes, APIs, DB logic
├── train_model.py          # ML model training + evaluation + charts
├── generate_dataset.py     # Synthetic dataset generator (1000 rows)
├── stress_dataset.csv      # Training dataset (7 features, 3 classes)
├── stress_model.pkl        # Trained ML model (auto-selected best)
├── label_encoder.pkl       # Label encoder for stress level classes
├── model_comparison.png    # Model accuracy comparison chart
├── confusion_matrix.png    # Confusion matrix of best model
├── mind stress.html        # Homepage — assessment form + results
├── insights.html           # Wellness insights dashboard
├── profile.html            # User profile page
├── login.html              # Login page
├── signup.html             # Registration page
├── .env                    # Environment variables (DB, Flask, Ollama)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## ⚙️ Setup & Installation

### 1. Prerequisites

- Python 3.9+
- MySQL Server running locally
- [Ollama](https://ollama.com) installed (optional — for AI advice feature)

### 2. Clone / Download the project

```bash
cd Desktop/stress
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Edit `.env` with your MySQL credentials:

```env
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=mindcheck

FLASK_SECRET_KEY=your-secret-key
FLASK_PORT=5000
FLASK_DEBUG=True

OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

### 5. (Optional) Generate dataset and retrain model

```bash
# Generate synthetic dataset
python generate_dataset.py

# Train and evaluate all models — saves best model as stress_model.pkl
python train_model.py
```

> The pre-trained `stress_model.pkl` and `label_encoder.pkl` are already included. Skip this step unless you want to retrain.

### 6. (Optional) Set up Ollama for AI advice

```bash
# Install Ollama from https://ollama.com, then pull the model
ollama pull llama3.2
ollama serve
```

### 7. Run the app

```bash
python app.py
```

Open your browser at **http://localhost:5000**

---

## 🗄️ Database

The app auto-creates the `mindcheck` database and all tables on first run. No manual SQL setup needed.

**Tables created automatically:**

```sql
users (
    id, first_name, last_name, email, username,
    password, age, occupation, avatar_url, created_at
)

stress_history (
    id, user_id, overall_score,
    emotional, sleep, overwhelm, mood,
    physical, concentration, irritability,
    stress_level, created_at
)
```

**Demo accounts seeded on first run:**

| Username | Password | Role |
|----------|----------|------|
| `admin` | `admin123` | Admin |
| `demo` | `demo123` | Demo user (with pre-seeded history) |
| `user` | `user123` | Test user |

---

## 🤖 ML Model

The training pipeline in `train_model.py` evaluates 5 classifiers and automatically selects the best:

| Model | Notes |
|-------|-------|
| Random Forest | 200 estimators |
| Gradient Boosting | 200 estimators |
| SVM (RBF kernel) | C=10, gamma=scale |
| K-Nearest Neighbors | k=7 |
| Logistic Regression | max_iter=1000 |

**Input features (7, scale 1–5):**

1. Nervousness / emotional stress frequency
2. Sleep quality
3. Task overwhelm frequency
4. Mood & outlook
5. Physical symptoms
6. Concentration ability
7. Irritability frequency

**Output classes:** `Low` · `Moderate` · `High`

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/login` | Authenticate user |
| `GET` | `/logout` | Clear session |
| `POST` | `/register` | Create new account |
| `POST` | `/predict` | Run stress prediction |
| `GET` | `/api/history` | Get user's assessment history |
| `GET` | `/api/insights` | Get AI-generated wellness insights |
| `POST` | `/api/ai-advice` | Get Ollama LLM advice |
| `GET` | `/api/profile` | Get user profile data |
| `POST` | `/api/profile/update` | Update user profile |

---

## 🎨 Design System

All pages share a unified design language:

- **Colors:** Primary `#6C63FF` · Accent `#FF6584` · Green `#10B981` · Yellow `#F59E0B` · Red `#EF4444`
- **Typography:** Nunito (headings, 900 weight) + DM Sans (body)
- **Cards:** 18px border radius, glassmorphism navbar, subtle gradient overlays
- **Animations:** GSAP scroll reveals, Three.js particle background, CSS keyframe transitions
- **Themes:** Full dark/light mode on all pages, persisted in `localStorage`

---

## 📄 License

This project is for educational purposes.

---

*Built with 💜 using Flask, scikit-learn, and Three.js*
