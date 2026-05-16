"""
Flask backend with MySQL database for user authentication,
ML-powered stress prediction, and AI wellness companion insights.
"""

from flask import Flask, request, jsonify, send_from_directory, session, redirect
from dotenv import load_dotenv
import mysql.connector
import bcrypt
import joblib
import numpy as np
import os
import random
from datetime import datetime, timedelta

# Load environment variables from .env
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model and label encoder
model = joblib.load(os.path.join(BASE_DIR, 'stress_model.pkl'))
le = joblib.load(os.path.join(BASE_DIR, 'label_encoder.pkl'))

app = Flask(__name__, static_folder=BASE_DIR)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'mindcheck-secret-key-2024')

# MySQL Configuration (loaded from .env)
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'mindcheck'),
}


def get_db():
    """Get a MySQL connection."""
    return mysql.connector.connect(**DB_CONFIG)


# ── Category metadata for the insights engine ────────────────
CATEGORY_META = {
    'emotional':      {'label': 'Emotional Wellbeing', 'icon': '😮'},
    'sleep':          {'label': 'Sleep Quality',       'icon': '🌙'},
    'overwhelm':      {'label': 'Task Overwhelm',     'icon': '🌊'},
    'mood':           {'label': 'Mood & Outlook',      'icon': '😔'},
    'physical':       {'label': 'Physical Symptoms',   'icon': '❤️'},
    'concentration':  {'label': 'Focus & Concentration','icon': '🎯'},
    'irritability':   {'label': 'Irritability',        'icon': '😤'},
}

# ── Recommendation bank per category ─────────────────────────
RECOMMENDATION_BANK = {
    'emotional': [
        {'title': 'Practice Emotional Check-ins', 'detail': 'Set 3 daily alarms to pause and label your current emotion — naming feelings reduces their intensity by up to 50%.'},
        {'title': 'Try Journaling', 'detail': 'Spend 5 minutes each evening writing about your day. Expressive writing helps process emotions and reduce anxiety.'},
        {'title': 'Box Breathing Technique', 'detail': 'When stress hits, breathe in for 4 counts, hold 4, exhale 4, hold 4. Repeat 3 cycles to calm your nervous system instantly.'},
    ],
    'sleep': [
        {'title': 'Create a Wind-Down Ritual', 'detail': 'Start dimming lights 1 hour before bed and avoid screens. A warm shower 90 minutes before sleep can improve sleep onset by 36%.'},
        {'title': 'Consistent Sleep Schedule', 'detail': 'Go to bed and wake up at the same time every day — even weekends. Your body\'s internal clock thrives on consistency.'},
        {'title': 'Optimize Your Sleep Space', 'detail': 'Keep your bedroom cool (18-20°C), dark, and quiet. Consider blackout curtains or a white noise app.'},
    ],
    'overwhelm': [
        {'title': 'The 2-Minute Rule', 'detail': 'If a task takes less than 2 minutes, do it immediately. This prevents small tasks from piling up and creating mental clutter.'},
        {'title': 'Time-Block Your Day', 'detail': 'Break your day into focused 25-minute blocks with 5-minute breaks (Pomodoro method). This makes large workloads feel manageable.'},
        {'title': 'Learn to Say No', 'detail': 'Practice setting boundaries by declining one non-essential commitment this week. Protecting your time is an act of self-care.'},
    ],
    'mood': [
        {'title': 'Gratitude Micro-Practice', 'detail': 'Each morning, write down 3 specific things you\'re grateful for. This rewires your brain to notice positives over time.'},
        {'title': 'Get Natural Sunlight', 'detail': 'Spend at least 15 minutes outside in natural light each morning. Sunlight boosts serotonin and helps regulate your mood.'},
        {'title': 'Connect with Someone', 'detail': 'Reach out to a friend or loved one today, even just a quick message. Social connection is one of the strongest mood boosters.'},
    ],
    'physical': [
        {'title': 'Gentle Movement Breaks', 'detail': 'Every 90 minutes, stand up and stretch for 2 minutes. Shoulder rolls and neck stretches can release tension and prevent headaches.'},
        {'title': 'Stay Hydrated', 'detail': 'Aim for 8 glasses of water daily. Dehydration can cause headaches, fatigue, and muscle tension that mimic stress symptoms.'},
        {'title': 'Progressive Muscle Relaxation', 'detail': 'Before bed, tense and release each muscle group for 5 seconds. Start from your toes and work up — this melts physical tension.'},
    ],
    'concentration': [
        {'title': 'Single-Tasking', 'detail': 'Close all unnecessary tabs and apps. Focus on one task at a time — multitasking reduces productivity by up to 40%.'},
        {'title': 'Mindful Focus Training', 'detail': 'Start with 5 minutes of mindfulness meditation daily. Apps like Headspace have guided sessions specifically for improving focus.'},
        {'title': 'Take Brain Breaks', 'detail': 'After 50 minutes of focused work, take a 10-minute break. Walk, look at nature, or do something non-screen-based to recharge.'},
    ],
    'irritability': [
        {'title': 'The STOP Technique', 'detail': 'When irritated: Stop, Take a breath, Observe your feelings without judgment, Proceed mindfully. This 10-second reset prevents reactive responses.'},
        {'title': 'Identify Your Triggers', 'detail': 'Keep a brief log of what triggers your irritability this week. Awareness of patterns is the first step to managing them.'},
        {'title': 'Physical Release', 'detail': 'Channel frustration into a brisk walk, quick workout, or even squeezing a stress ball. Physical activity metabolizes stress hormones.'},
    ],
}

# ── Motivation message templates ─────────────────────────────
MOTIVATION_TEMPLATES = {
    'improving': [
        "Your scores show real progress — every small step you've taken is adding up, and you should feel proud of the momentum you're building.",
        "You're moving in a wonderful direction — your recent scores reflect genuine improvement, and that takes real effort and courage.",
        "The positive trend in your journey is no accident — it's the result of your commitment to yourself, and that's truly inspiring.",
    ],
    'stable': [
        "Consistency is a strength — maintaining your current level shows resilience, and every day you show up for yourself matters.",
        "Staying steady in a busy world is an achievement in itself — you're doing better than you think, and small improvements compound over time.",
        "Your steady path shows inner strength — remember that progress isn't always dramatic, and you're building a solid foundation for wellness.",
    ],
    'worsening': [
        "Tough stretches are part of everyone's journey — recognizing where you are right now is a brave and important first step toward feeling better.",
        "It's okay to have difficult periods — the fact that you're checking in with yourself shows incredible self-awareness, and brighter days are ahead.",
        "Be gentle with yourself during this time — seeking to understand your stress is already an act of self-care, and you don't have to face this alone.",
    ],
    'first_time': [
        "Welcome to your wellness journey! Taking this first assessment shows you care about your mental health — that's the most important step of all.",
        "You've just taken a meaningful step by checking in with yourself — keep coming back, and you'll build a clear picture of your wellness over time.",
    ],
}


def init_db():
    """Create the database and users table if they don't exist."""
    # First connect without database to create it
    conn = mysql.connector.connect(
        host=DB_CONFIG['host'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password'],
    )
    cursor = conn.cursor()
    cursor.execute("CREATE DATABASE IF NOT EXISTS mindcheck")
    cursor.close()
    conn.close()

    # Now connect to the database and create the tables
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            first_name VARCHAR(100) NOT NULL,
            last_name VARCHAR(100) NOT NULL,
            email VARCHAR(255) NOT NULL UNIQUE,
            username VARCHAR(100) NOT NULL UNIQUE,
            password VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Add profile columns if they don't exist
    for col, coldef in [('age', 'INT DEFAULT NULL'), ('occupation', 'VARCHAR(150) DEFAULT NULL'), ('avatar_url', 'VARCHAR(500) DEFAULT NULL')]:
        try:
            cursor.execute(f"ALTER TABLE users ADD COLUMN {col} {coldef}")
        except mysql.connector.Error:
            pass  # Column already exists

    # Add UNIQUE constraint on email if not already present (for existing databases)
    try:
        cursor.execute("ALTER TABLE users ADD CONSTRAINT uq_users_email UNIQUE (email)")
        conn.commit()
    except mysql.connector.Error:
        pass  # Constraint already exists

    # Contact messages table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS contact_messages (
            id         INT AUTO_INCREMENT PRIMARY KEY,
            name       VARCHAR(150) NOT NULL,
            email      VARCHAR(255) NOT NULL,
            subject    VARCHAR(255) DEFAULT '',
            message    TEXT NOT NULL,
            user_id    INT DEFAULT NULL,
            is_read    TINYINT(1) DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
        )
    """)
    conn.commit()

    # Stress history table for longitudinal tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stress_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            overall_score INT NOT NULL,
            emotional INT DEFAULT 0,
            sleep INT DEFAULT 0,
            overwhelm INT DEFAULT 0,
            mood INT DEFAULT 0,
            physical INT DEFAULT 0,
            concentration INT DEFAULT 0,
            irritability INT DEFAULT 0,
            stress_level VARCHAR(20) DEFAULT 'Unknown',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)
    conn.commit()

    # Insert demo users if table is empty
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]
    if count == 0:
        demo_users = [
            ('Admin', 'User', 'admin@mindcheck.com', 'admin',
             bcrypt.hashpw(b'admin123', bcrypt.gensalt()).decode()),
            ('Demo', 'User', 'demo@mindcheck.com', 'demo',
             bcrypt.hashpw(b'demo123', bcrypt.gensalt()).decode()),
            ('Test', 'User', 'user@mindcheck.com', 'user',
             bcrypt.hashpw(b'user123', bcrypt.gensalt()).decode()),
        ]
        cursor.executemany(
            "INSERT INTO users (first_name, last_name, email, username, password) VALUES (%s, %s, %s, %s, %s)",
            demo_users
        )
        conn.commit()
        print("[DB] Inserted 3 demo users (admin/admin123, demo/demo123, user/user123)")

    # Seed demo stress history for user id=2 (demo user) if empty
    cursor.execute("SELECT COUNT(*) FROM stress_history")
    history_count = cursor.fetchone()[0]
    if history_count == 0:
        _seed_demo_history(cursor, conn)

    cursor.close()
    conn.close()
    print("[DB] MySQL database 'mindcheck' ready.")


def _seed_demo_history(cursor, conn):
    """Seed 18 demo stress history entries spanning 30 days for the demo user."""
    # Get demo user id
    cursor.execute("SELECT id FROM users WHERE username = 'demo'")
    row = cursor.fetchone()
    if not row:
        return
    demo_id = row[0]

    now = datetime.now()
    # Create a realistic pattern: starting high, gradually improving
    entries = []
    base_scores = [
        # Day offset, overall, emotional, sleep, overwhelm, mood, physical, concentration, irritability
        (-30, 72, 80, 75, 85, 65, 60, 70, 75),
        (-28, 68, 75, 70, 80, 60, 65, 68, 72),
        (-25, 74, 82, 72, 78, 68, 70, 72, 78),
        (-22, 65, 70, 68, 75, 58, 55, 65, 68),
        (-20, 62, 68, 65, 72, 55, 58, 60, 65),
        (-18, 58, 62, 60, 68, 50, 52, 58, 62),
        (-15, 55, 58, 55, 65, 48, 50, 55, 58),
        (-13, 60, 65, 58, 70, 52, 55, 58, 62),
        (-11, 52, 55, 50, 60, 45, 48, 52, 55),
        (-9,  48, 50, 45, 55, 42, 45, 48, 50),
        (-7,  50, 52, 48, 58, 44, 42, 50, 52),
        (-6,  45, 48, 42, 52, 38, 40, 45, 48),
        (-5,  42, 45, 40, 48, 35, 38, 42, 45),
        (-4,  40, 42, 38, 45, 35, 36, 40, 42),
        (-3,  38, 40, 35, 42, 32, 35, 38, 40),
        (-2,  35, 38, 32, 40, 30, 32, 35, 38),
        (-1,  33, 35, 30, 38, 28, 30, 33, 35),
        (0,   30, 32, 28, 35, 25, 28, 30, 32),
    ]

    for entry in entries if False else base_scores:
        day_offset = entry[0]
        ts = now + timedelta(days=day_offset, hours=random.randint(8, 20),
                             minutes=random.randint(0, 59))
        overall = entry[1]
        level = 'High' if overall >= 67 else ('Moderate' if overall >= 34 else 'Low')

        cursor.execute(
            """INSERT INTO stress_history
               (user_id, overall_score, emotional, sleep, overwhelm, mood,
                physical, concentration, irritability, stress_level, created_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (demo_id, overall, entry[2], entry[3], entry[4], entry[5],
             entry[6], entry[7], entry[8], level, ts)
        )
    conn.commit()
    print(f"[DB] Seeded {len(base_scores)} demo stress history entries for 'demo' user.")


# ── Login page ────────────────────────────────────────────────
@app.route('/login-page')
def login_page():
    if session.get('logged_in'):
        return redirect('/')
    return send_from_directory(BASE_DIR, 'login.html')


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json(force=True)
    username = data.get('username', '').strip()
    password = data.get('password', '')

    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
    except Exception as e:
        return jsonify({'success': False, 'message': f'Database error: {str(e)}'})

    if user and bcrypt.checkpw(password.encode(), user['password'].encode()):
        session['logged_in'] = True
        session['username'] = username
        session['user_id'] = user['id']
        return jsonify({'success': True})

    return jsonify({'success': False, 'message': 'Invalid username or password.'})


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login-page')


# ── Home page (requires login) ───────────────────────────────
@app.route('/')
def index():
    if not session.get('logged_in'):
        return redirect('/login-page')
    return send_from_directory(BASE_DIR, 'mind stress.html')


# ── Insights page (requires login) ───────────────────────────
@app.route('/insights')
def insights_page():
    if not session.get('logged_in'):
        return redirect('/login-page')
    return send_from_directory(BASE_DIR, 'insights.html')


# ── Profile page (requires login) ────────────────────────────
@app.route('/profile')
def profile_page():
    if not session.get('logged_in'):
        return redirect('/login-page')
    return send_from_directory(BASE_DIR, 'profile.html')


# ── Profile API ──────────────────────────────────────────────
@app.route('/api/profile')
def get_profile():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """SELECT id, first_name, last_name, email, username, age,
                      occupation, avatar_url, created_at
               FROM users WHERE id = %s""", (user_id,))
        user = cursor.fetchone()
        # Get assessment count
        cursor.execute("SELECT COUNT(*) as cnt FROM stress_history WHERE user_id = %s", (user_id,))
        stats = cursor.fetchone()
        cursor.close()
        conn.close()
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    if not user:
        return jsonify({'error': 'User not found'}), 404
    return jsonify({
        'id': user['id'],
        'first_name': user['first_name'] or '',
        'last_name': user['last_name'] or '',
        'email': user['email'] or '',
        'username': user['username'] or '',
        'age': user['age'],
        'occupation': user['occupation'] or '',
        'avatar_url': user['avatar_url'] or '',
        'created_at': user['created_at'].strftime('%B %d, %Y') if user['created_at'] else '',
        'assessments': stats['cnt'] if stats else 0,
    })


@app.route('/api/profile/update', methods=['POST'])
def update_profile():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401
    data = request.get_json(force=True)
    allowed = {'first_name', 'last_name', 'email', 'age', 'occupation', 'avatar_url'}
    updates = {k: v for k, v in data.items() if k in allowed}
    if not updates:
        return jsonify({'error': 'No valid fields to update'}), 400
    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)

        # If email is being updated, check it isn't already taken by another account
        if 'email' in updates and updates['email']:
            cursor.execute(
                "SELECT id FROM users WHERE email = %s AND id != %s",
                (updates['email'], user_id)
            )
            rows = cursor.fetchall()  # consume all rows
            if rows:
                cursor.close()
                conn.close()
                return jsonify({'error': 'This email is already used by another account.'}), 409

        set_clause = ', '.join(f"{k} = %s" for k in updates)
        vals = list(updates.values()) + [user_id]
        cursor.execute(f"UPDATE users SET {set_clause} WHERE id = %s", vals)
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({'success': True})
    except mysql.connector.IntegrityError:
        return jsonify({'error': 'This email is already used by another account.'}), 409
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Prediction endpoint (now saves to history) ───────────────
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    answers = data.get('answers', [])

    if len(answers) != 7:
        return jsonify({'error': 'Exactly 7 answers required'}), 400

    features = [int(a) + 1 for a in answers]

    prediction = model.predict([features])[0]
    stress_label = le.inverse_transform([prediction])[0]

    avg = np.mean(features)
    overall_score = int(round((avg - 1) / 4 * 100))

    feature_keys = ['emotional', 'sleep', 'overwhelm', 'mood', 'physical', 'concentration', 'irritability']
    feature_scores = {}
    for key, val in zip(feature_keys, features):
        feature_scores[key] = int(round((val - 1) / 4 * 100))

    # Save to stress_history if user is logged in
    user_id = session.get('user_id')
    if user_id:
        try:
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO stress_history
                   (user_id, overall_score, emotional, sleep, overwhelm,
                    mood, physical, concentration, irritability, stress_level)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (user_id, overall_score,
                 feature_scores['emotional'], feature_scores['sleep'],
                 feature_scores['overwhelm'], feature_scores['mood'],
                 feature_scores['physical'], feature_scores['concentration'],
                 feature_scores['irritability'], stress_label)
            )
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"[WARN] Failed to save stress history: {e}")

    return jsonify({
        'stress_level': stress_label,
        'score': overall_score,
        'feature_scores': feature_scores
    })


# ── Stress History API (for charts) ──────────────────────────
@app.route('/api/history')
def get_history():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """SELECT overall_score, emotional, sleep, overwhelm, mood,
                      physical, concentration, irritability, stress_level,
                      created_at
               FROM stress_history
               WHERE user_id = %s
               ORDER BY created_at ASC""",
            (user_id,)
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    history = []
    for row in rows:
        history.append({
            'score': row['overall_score'],
            'emotional': row['emotional'],
            'sleep': row['sleep'],
            'overwhelm': row['overwhelm'],
            'mood': row['mood'],
            'physical': row['physical'],
            'concentration': row['concentration'],
            'irritability': row['irritability'],
            'level': row['stress_level'],
            'date': row['created_at'].strftime('%Y-%m-%d %H:%M') if row['created_at'] else '',
        })

    return jsonify({'history': history})


# ── AI Wellness Companion Insights API ───────────────────────
@app.route('/api/insights')
def get_insights():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """SELECT overall_score, emotional, sleep, overwhelm, mood,
                      physical, concentration, irritability, stress_level,
                      created_at
               FROM stress_history
               WHERE user_id = %s
               ORDER BY created_at ASC""",
            (user_id,)
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    if not rows:
        return jsonify({
            'trend': 'stable',
            'summary': "You haven't taken any assessments yet. Take your first stress check to start tracking your wellness journey!",
            'topStressors': [],
            'recommendations': [],
            'motivation': random.choice(MOTIVATION_TEMPLATES['first_time']),
            'empty': True,
        })

    # ── Compute trend ────────────────────────────────────────
    scores = [r['overall_score'] for r in rows]
    n = len(scores)

    if n == 1:
        trend = 'stable'
        trend_detail = 'first_time'
    else:
        # Compare recent half vs older half
        mid = max(1, n // 2)
        older_avg = sum(scores[:mid]) / mid
        recent_avg = sum(scores[mid:]) / (n - mid)
        diff = recent_avg - older_avg

        if diff <= -5:
            trend = 'improving'
        elif diff >= 5:
            trend = 'worsening'
        else:
            trend = 'stable'
        trend_detail = trend

    # ── Identify top stressors ───────────────────────────────
    categories = ['emotional', 'sleep', 'overwhelm', 'mood', 'physical', 'concentration', 'irritability']
    cat_averages = {}
    for cat in categories:
        vals = [r[cat] for r in rows if r[cat] is not None]
        cat_averages[cat] = sum(vals) / len(vals) if vals else 0

    sorted_cats = sorted(cat_averages, key=cat_averages.get, reverse=True)
    top_stressors = [CATEGORY_META[c]['label'] for c in sorted_cats[:3]]
    top_cat_keys = sorted_cats[:3]

    # ── Generate summary ─────────────────────────────────────
    latest = rows[-1]
    latest_score = latest['overall_score']
    latest_level = latest['stress_level']

    if n == 1:
        summary = (
            f"Your first assessment shows a stress score of {latest_score}/100, "
            f"which falls in the {latest_level.lower()} range. "
            f"Keep taking assessments to build a clear picture of your stress patterns over time."
        )
    else:
        first_score = rows[0]['overall_score']
        overall_change = latest_score - first_score
        duration_days = max(1, (rows[-1]['created_at'] - rows[0]['created_at']).days)

        top1_label = CATEGORY_META[top_cat_keys[0]]['label']
        top1_avg = round(cat_averages[top_cat_keys[0]])

        if trend == 'improving':
            summary = (
                f"Over the past {duration_days} days across {n} assessments, your stress has been trending downward — "
                f"your score has dropped by {abs(overall_change)} points, which is great progress. "
                f"Your highest area of concern remains {top1_label} (avg {top1_avg}/100), "
                f"so focusing there could help you continue improving."
            )
        elif trend == 'worsening':
            summary = (
                f"Over the past {duration_days} days, your stress levels have been gradually increasing — "
                f"your score has risen by {abs(overall_change)} points across {n} assessments. "
                f"{top1_label} appears to be your biggest contributor at an average of {top1_avg}/100, "
                f"and addressing it could make a meaningful difference."
            )
        else:
            summary = (
                f"Your stress has been relatively stable over the past {duration_days} days across {n} assessments, "
                f"hovering around a score of {round(sum(scores) / n)}. "
                f"{top1_label} consistently ranks as your top stressor (avg {top1_avg}/100), "
                f"so targeted strategies there could help shift things in a positive direction."
            )

    # ── Generate recommendations ─────────────────────────────
    recommendations = []
    for cat_key in top_cat_keys:
        tips = RECOMMENDATION_BANK.get(cat_key, [])
        if tips:
            tip = random.choice(tips)
            recommendations.append({
                'title': tip['title'],
                'detail': tip['detail'],
            })

    # Pad to 3 if needed
    while len(recommendations) < 3:
        all_tips = [t for bank in RECOMMENDATION_BANK.values() for t in bank]
        tip = random.choice(all_tips)
        if tip not in recommendations:
            recommendations.append({'title': tip['title'], 'detail': tip['detail']})

    # ── Motivation message ───────────────────────────────────
    motivation = random.choice(MOTIVATION_TEMPLATES.get(trend_detail if n == 1 else trend, MOTIVATION_TEMPLATES['stable']))

    return jsonify({
        'trend': trend,
        'summary': summary,
        'topStressors': top_stressors,
        'recommendations': recommendations[:3],
        'motivation': motivation,
        'empty': False,
        'totalAssessments': n,
        'latestScore': latest_score,
        'latestLevel': latest_level,
    })


# ── Ollama AI Advice API ─────────────────────────────────────
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')


@app.route('/api/ai-advice', methods=['POST'])
def ai_advice():
    """Generate personalized wellness advice using Ollama."""
    import requests as http_requests

    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Not authenticated'}), 401

    # Fetch latest stress record
    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """SELECT overall_score, emotional, sleep, overwhelm, mood,
                      physical, concentration, irritability, stress_level
               FROM stress_history
               WHERE user_id = %s
               ORDER BY created_at DESC LIMIT 1""",
            (user_id,)
        )
        latest = cursor.fetchone()
        cursor.close()
        conn.close()
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500

    if not latest:
        return jsonify({'error': 'No assessment data found. Take an assessment first.'}), 404

    # Build the prompt
    prompt = f"""You are a compassionate and knowledgeable wellness advisor for a mental health app called MindCheck.

Based on the user's stress assessment scores below, provide exactly 3 specific, actionable wellness tips.

Stress Profile (0 = no stress, 100 = maximum stress):
- Emotional Wellbeing: {latest['emotional']}/100
- Sleep Quality: {latest['sleep']}/100
- Task Overwhelm: {latest['overwhelm']}/100
- Mood & Outlook: {latest['mood']}/100
- Physical Symptoms: {latest['physical']}/100
- Concentration: {latest['concentration']}/100
- Irritability: {latest['irritability']}/100
- Overall Stress Level: {latest['stress_level']} ({latest['overall_score']}/100)

Rules:
1. Focus your advice on the TOP 2 highest-scoring (most stressed) areas
2. Each tip must be practical and doable TODAY
3. Be warm and encouraging, not clinical
4. Keep each tip to 2-3 sentences maximum
5. Format: Start each tip with a clear action title

Provide your 3 tips now:"""

    # Call Ollama
    try:
        resp = http_requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                'model': OLLAMA_MODEL,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.7,
                    'num_predict': 400,
                }
            },
            timeout=60,
        )
        resp.raise_for_status()
        result = resp.json()
        advice_text = result.get('response', '').strip()

        if not advice_text:
            return jsonify({'error': 'AI returned an empty response. Please try again.'}), 500

        return jsonify({'advice': advice_text})

    except http_requests.exceptions.ConnectionError:
        return jsonify({'error': 'Ollama is not running. Please start Ollama and try again.'}), 503
    except http_requests.exceptions.Timeout:
        return jsonify({'error': 'AI took too long to respond. Please try again.'}), 504
    except Exception as e:
        return jsonify({'error': f'AI service error: {str(e)}'}), 500


# ── Contact Message API ──────────────────────────────────────
@app.route('/api/contact', methods=['POST'])
def submit_contact():
    data = request.get_json(force=True)
    name    = data.get('name', '').strip()
    email   = data.get('email', '').strip()
    subject = data.get('subject', '').strip()
    message = data.get('message', '').strip()

    # Validation
    if not name or not email or not message:
        return jsonify({'success': False, 'message': 'Name, email, and message are required.'}), 400
    if len(name) > 150:
        return jsonify({'success': False, 'message': 'Name is too long (max 150 characters).'}), 400
    if len(subject) > 255:
        return jsonify({'success': False, 'message': 'Subject is too long (max 255 characters).'}), 400
    if len(message) > 5000:
        return jsonify({'success': False, 'message': 'Message is too long (max 5000 characters).'}), 400
    if '@' not in email or '.' not in email:
        return jsonify({'success': False, 'message': 'Please enter a valid email address.'}), 400

    # Attach logged-in user_id if available
    user_id = session.get('user_id')

    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO contact_messages (name, email, subject, message, user_id)
               VALUES (%s, %s, %s, %s, %s)""",
            (name, email, subject, message, user_id)
        )
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({'success': True, 'message': 'Your message has been received! We\'ll get back to you soon.'})
    except Exception as e:
        return jsonify({'success': False, 'message': 'Something went wrong. Please try again.'}), 500


# ── Register ──────────────────────────────────────────────────
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return send_from_directory(BASE_DIR, 'signup.html')

    data = request.get_json(force=True)
    username = data.get('username', '').strip()
    password = data.get('password', '')
    email = data.get('email', '').strip()
    first_name = data.get('first_name', '').strip()
    last_name = data.get('last_name', '').strip()

    if not username or not password or not email:
        return jsonify({'success': False, 'message': 'All fields are required.'})
    if len(password) < 6:
        return jsonify({'success': False, 'message': 'Password must be at least 6 characters.'})

    # Hash the password
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)

        # Check for duplicate username or email before inserting
        cursor.execute(
            "SELECT id, username, email FROM users WHERE username = %s OR email = %s",
            (username, email)
        )
        existing_rows = cursor.fetchall()  # consume ALL rows to avoid "Unread result found"

        if existing_rows:
            cursor.close()
            conn.close()
            # Check what exactly matched
            for row in existing_rows:
                if row['username'] == username:
                    return jsonify({'success': False, 'message': 'Username already taken. Please choose a different username.'})
            return jsonify({'success': False, 'message': 'An account with this email already exists. Please log in or use a different email.'})

        cursor.execute(
            "INSERT INTO users (first_name, last_name, email, username, password) VALUES (%s, %s, %s, %s, %s)",
            (first_name, last_name, email, username, hashed)
        )
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({'success': True})
    except mysql.connector.IntegrityError as e:
        err = str(e).lower()
        if 'email' in err:
            return jsonify({'success': False, 'message': 'An account with this email already exists. Please log in or use a different email.'})
        return jsonify({'success': False, 'message': 'Username already taken. Please choose a different username.'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Something went wrong. Please try again.'})

# ── /docs — Live API status dashboard ───────────────────────
@app.route('/docs')
def docs():
    import platform, sys

    # Test DB connection
    db_status = 'connected'
    db_error  = ''
    user_count = 0
    assessment_count = 0
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM users")
        user_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM stress_history")
        assessment_count = cur.fetchone()[0]
        cur.close(); conn.close()
    except Exception as e:
        db_status = 'error'
        db_error  = str(e)

    # Test Ollama
    ollama_status = 'unknown'
    try:
        import requests as _r
        r = _r.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        ollama_status = 'running' if r.status_code == 200 else f'error ({r.status_code})'
    except Exception:
        ollama_status = 'offline'

    db_dot = '#10B981' if db_status == 'connected' else '#EF4444'
    ol_dot = '#10B981' if ollama_status == 'running' else '#F59E0B' if ollama_status == 'unknown' else '#EF4444'

    routes = [
        ('GET',  '/',                   'Home — Stress Assessment page',          True),
        ('GET',  '/insights',           'Wellness Insights dashboard',             True),
        ('GET',  '/profile',            'User Profile page',                       True),
        ('GET',  '/login-page',         'Login page',                              False),
        ('GET',  '/register',           'Register / Sign Up page',                 False),
        ('POST', '/login',              'Authenticate user — returns JSON',        False),
        ('GET',  '/logout',             'Clear session — redirect to login',       False),
        ('POST', '/register',           'Create new account — returns JSON',       False),
        ('POST', '/predict',            'Run ML stress prediction — JSON result',  True),
        ('GET',  '/api/history',        'Get user assessment history',             True),
        ('GET',  '/api/insights',       'Get AI wellness insights',                True),
        ('POST', '/api/ai-advice',      'Get Ollama LLM personalised advice',      True),
        ('GET',  '/api/profile',        'Get logged-in user profile data',         True),
        ('POST', '/api/profile/update', 'Update user profile fields',              True),
        ('GET',  '/docs',               'This page — live API status dashboard',   False),
        ('POST', '/api/contact',        'Submit a Get In Touch message',           False),
    ]

    rows = ''
    for method, path, desc, auth in routes:
        mc = {'GET': '#10B981', 'POST': '#6C63FF'}.get(method, '#6B7280')
        auth_badge = (
            '<span style="background:rgba(245,158,11,.15);color:#F59E0B;border-radius:99px;'
            'padding:2px 10px;font-size:.75rem;font-weight:700">&#128274; Auth</span>'
            if auth else
            '<span style="background:rgba(16,185,129,.12);color:#10B981;border-radius:99px;'
            'padding:2px 10px;font-size:.75rem;font-weight:700">Public</span>'
        )
        link = (f'<a href="{path}" target="_blank" style="color:#6C63FF;text-decoration:none;'
                f'font-weight:700;font-family:monospace">{path}</a>'
                if method == 'GET' else
                f'<span style="font-family:monospace">{path}</span>')
        rows += (
            f'<tr>'
            f'<td><span style="background:{mc};color:#fff;border-radius:6px;padding:3px 10px;'
            f'font-size:.78rem;font-weight:800;font-family:monospace">{method}</span></td>'
            f'<td>{link}</td>'
            f'<td style="color:var(--muted);font-size:.85rem">{desc}</td>'
            f'<td>{auth_badge}</td>'
            f'</tr>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>MindCheck \u2014 API Docs</title>
<link href="https://fonts.googleapis.com/css2?family=Nunito:wght@700;800;900&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet"/>
<style>
:root{{--p:#6C63FF;--p3:#4A42CC;--bg:#F4F5FF;--card:#FFFFFF;--text:#1A1A2E;--muted:#6B7280;--border:#E5E7EB;--sh:0 4px 30px rgba(108,99,255,.1);--sh2:0 12px 40px rgba(108,99,255,.22);--r:18px;--r2:12px}}
[data-theme=dark]{{--bg:#080812;--card:#13132A;--text:#EEEEFF;--muted:#9CA3AF;--border:#252540;--sh:0 4px 30px rgba(0,0,0,.5)}}
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:"DM Sans",sans-serif;background:var(--bg);color:var(--text);min-height:100vh;padding:2rem;transition:background .3s,color .3s}}
.wrap{{max-width:1000px;margin:0 auto;display:flex;flex-direction:column;gap:1.4rem}}
.card{{background:var(--card);border-radius:var(--r);padding:1.8rem;box-shadow:var(--sh);border:1px solid var(--border)}}
.badge{{display:inline-flex;align-items:center;gap:.4rem;background:rgba(108,99,255,.08);border:1px solid rgba(108,99,255,.2);color:var(--p);font-size:.78rem;font-weight:700;padding:.3rem .9rem;border-radius:99px;margin-bottom:.8rem}}
h1{{font-family:"Nunito",sans-serif;font-weight:900;font-size:2rem;letter-spacing:-1px;margin-bottom:.3rem}}
h1 span{{background:linear-gradient(135deg,#6C63FF,#FF6584);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}}
.sub{{color:var(--muted);font-size:.95rem;margin-bottom:1.4rem;line-height:1.6}}
.status-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem}}
.stat{{background:var(--bg);border:1px solid var(--border);border-radius:var(--r2);padding:1.2rem;text-align:center;transition:transform .25s,box-shadow .25s}}
.stat:hover{{transform:translateY(-4px);box-shadow:var(--sh2)}}
.stat-num{{font-family:"Nunito",sans-serif;font-weight:900;font-size:1.8rem;color:var(--p)}}
.stat-label{{font-size:.78rem;color:var(--muted);margin-top:.2rem}}
.dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:.4rem;vertical-align:middle}}
table{{width:100%;border-collapse:collapse}}
th{{text-align:left;font-size:.75rem;font-weight:700;color:var(--muted);padding:.6rem 1rem;border-bottom:2px solid var(--border);text-transform:uppercase;letter-spacing:.5px}}
td{{padding:.75rem 1rem;border-bottom:1px solid var(--border);vertical-align:middle}}
tr:last-child td{{border-bottom:none}}
tr:hover td{{background:rgba(108,99,255,.03)}}
.section-title{{font-family:"Nunito",sans-serif;font-weight:800;font-size:1.05rem;margin-bottom:1rem}}
.info-grid{{display:grid;grid-template-columns:1fr 1fr;gap:.7rem}}
.info-row{{display:flex;justify-content:space-between;align-items:center;padding:.6rem .9rem;background:var(--bg);border-radius:var(--r2);font-size:.85rem}}
.info-key{{color:var(--muted);font-weight:600}}
.info-val{{font-weight:700;color:var(--text);font-family:monospace;font-size:.82rem}}
.back-btn{{display:inline-flex;align-items:center;gap:.5rem;background:var(--p);color:#fff;border:none;border-radius:var(--r2);padding:.5rem 1.2rem;font-family:"Nunito",sans-serif;font-weight:800;font-size:.85rem;text-decoration:none;transition:background .2s,transform .2s;margin-right:.6rem}}
.back-btn:hover{{background:var(--p3);transform:translateY(-1px)}}
.theme-btn{{position:fixed;top:1.2rem;right:1.5rem;width:42px;height:24px;border-radius:12px;border:none;cursor:pointer;background:#E5E7EB;transition:background .3s}}
.theme-btn.dark{{background:#6C63FF}}
.theme-btn::after{{content:"";position:absolute;top:3px;left:3px;width:18px;height:18px;border-radius:50%;background:white;transition:transform .3s;box-shadow:0 1px 4px rgba(0,0,0,.2)}}
.theme-btn.dark::after{{transform:translateX(18px)}}
code{{background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:1px 6px;font-size:.82rem}}
@media(max-width:700px){{.status-grid{{grid-template-columns:1fr 1fr}}.info-grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<button class="theme-btn" id="tb"></button>
<div class="wrap">
  <div class="card">
    <div class="badge">&#128214; Live API Docs</div>
    <h1>MindCheck <span>API Status</span></h1>
    <p class="sub">All services on a <strong>single Flask process</strong> at
      <code>http://localhost:5000</code> &mdash; frontend pages and backend API share the same port.</p>
    <a href="/" class="back-btn">&#127968; Home</a>
    <a href="/login-page" class="back-btn" style="background:#10B981">&#128274; Login</a>
  </div>
  <div class="status-grid">
    <div class="stat"><div class="stat-num">5000</div><div class="stat-label">&#127760; Port</div></div>
    <div class="stat">
      <div class="stat-num" style="color:{db_dot};font-size:1.3rem">
        <span class="dot" style="background:{db_dot}"></span>{'OK' if db_status == 'connected' else 'ERR'}
      </div>
      <div class="stat-label">&#128451; MySQL</div>
    </div>
    <div class="stat"><div class="stat-num">{user_count}</div><div class="stat-label">&#128100; Users</div></div>
    <div class="stat"><div class="stat-num">{assessment_count}</div><div class="stat-label">&#128203; Assessments</div></div>
  </div>
  <div class="card">
    <div class="section-title">&#9881;&#65039; Server Info</div>
    <div class="info-grid">
      <div class="info-row"><span class="info-key">Base URL</span><span class="info-val">http://localhost:5000</span></div>
      <div class="info-row"><span class="info-key">Python</span><span class="info-val">{sys.version.split()[0]}</span></div>
      <div class="info-row"><span class="info-key">Platform</span><span class="info-val">{platform.system()} {platform.release()}</span></div>
      <div class="info-row"><span class="info-key">Debug Mode</span><span class="info-val">{str(app.debug)}</span></div>
      <div class="info-row"><span class="info-key">Database</span><span class="info-val">{DB_CONFIG['database']}@{DB_CONFIG['host']}</span></div>
      <div class="info-row"><span class="info-key">DB Status</span>
        <span class="info-val" style="color:{db_dot}"><span class="dot" style="background:{db_dot}"></span>{db_status}{' &mdash; ' + db_error if db_error else ''}</span></div>
      <div class="info-row"><span class="info-key">Ollama URL</span><span class="info-val">{OLLAMA_URL}</span></div>
      <div class="info-row"><span class="info-key">Ollama Model</span><span class="info-val">{OLLAMA_MODEL}</span></div>
      <div class="info-row"><span class="info-key">Ollama</span>
        <span class="info-val" style="color:{ol_dot}"><span class="dot" style="background:{ol_dot}"></span>{ollama_status}</span></div>
      <div class="info-row"><span class="info-key">ML Model</span><span class="info-val">stress_model.pkl &#9989;</span></div>
    </div>
  </div>
  <div class="card">
    <div class="section-title">&#128268; All Routes &mdash; {len(routes)} endpoints</div>
    <table>
      <thead><tr><th>Method</th><th>Path</th><th>Description</th><th>Access</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
  <div class="card">
    <div class="section-title">&#128203; Notes</div>
    <p style="font-size:.88rem;color:var(--muted);line-height:1.75">
      &#128274; Endpoints marked <strong style="color:#F59E0B">Auth</strong> require an active login session.
      Hitting them directly in the browser returns <code>401</code> &mdash; that is correct.<br/>
      &#128161; Log in first at <a href="/login-page" style="color:#6C63FF;font-weight:700">/login-page</a> and all protected routes work normally.<br/>
      &#128683; <code>POST</code> endpoints cannot be opened in the browser &mdash; use the app UI or Postman.
    </p>
  </div>
</div>
<script>
(function(){{
  var btn=document.getElementById('tb');
  var saved=localStorage.getItem('mc-theme')||'light';
  document.documentElement.setAttribute('data-theme',saved);
  btn.classList.toggle('dark',saved==='dark');
  btn.addEventListener('click',function(){{
    var cur=document.documentElement.getAttribute('data-theme');
    var next=cur==='dark'?'light':'dark';
    document.documentElement.setAttribute('data-theme',next);
    localStorage.setItem('mc-theme',next);
    btn.classList.toggle('dark',next==='dark');
  }});
}})();
</script>
</body></html>"""
    return html


if __name__ == '__main__':
    init_db()
    print("=" * 50)
    print("MindCheck ML Backend  →  http://localhost:5000")
    print("API Docs              →  http://localhost:5000/docs")
    print("=" * 50)
    app.run(debug=True, port=5000)
