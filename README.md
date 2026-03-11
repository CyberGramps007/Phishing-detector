# 🛡️ Phishing Detector
**By Olabunmi Peace** | TF-IDF + Logistic Regression

A real-time phishing email detector built with a TF-IDF vectorizer and
logistic regression classifier — all running in the browser, no server needed.

---

## 🚀 Deploy to Vercel in 5 Minutes

### Step 1 — Install Node.js
Download from https://nodejs.org (pick the LTS version) and install it.

### Step 2 — Install Git
Download from https://git-scm.com and install it.

### Step 3 — Open a terminal / command prompt
On Windows: press `Win + R`, type `cmd`, press Enter.
On Mac: press `Cmd + Space`, type `Terminal`, press Enter.

### Step 4 — Navigate to this folder
```
cd path/to/phishing-detector
```
(e.g. `cd C:\Users\YourName\Downloads\phishing-detector`)

### Step 5 — Install dependencies
```
npm install
```
Wait for it to finish (about 1–2 minutes).

### Step 6 — Test it locally first
```
npm start
```
It will open http://localhost:3000 in your browser. Make sure it works!
Press `Ctrl + C` to stop it when done.

### Step 7 — Create a GitHub account
Go to https://github.com and sign up (free).

### Step 8 — Create a new GitHub repository
- Click the green "New" button
- Name it: `phishing-detector`
- Leave it Public
- Click "Create repository"

### Step 9 — Push your code to GitHub
Run these commands one by one (replace YOUR_USERNAME with your GitHub username):
```
git init
git add .
git commit -m "Initial commit — Phishing Detector by Olabunmi Peace"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/phishing-detector.git
git push -u origin main
```

### Step 10 — Deploy on Vercel
1. Go to https://vercel.com and sign up with your GitHub account
2. Click "Add New Project"
3. Find and click "phishing-detector" from your GitHub repos
4. Click "Deploy" — no settings to change, it auto-detects everything!
5. Wait ~60 seconds...
6. 🎉 Your live URL is ready! e.g. `https://phishing-detector-xyz.vercel.app`

---

## 🖥️ Run Locally (No Deployment)
```
npm install
npm start
```

---

## 🧠 How It Works
- **TF-IDF Vectorizer**: Converts text into numerical features based on term frequency × inverse document frequency
- **Logistic Regression**: Multiplies each feature by a learned weight, sums them with a bias, then applies sigmoid to get a probability
- **68-term vocabulary**: Covers phishing signals (e.g. "verify your account") and legitimate signals (e.g. "please find attached")

---

Built by **Olabunmi Peace** 🛡️ | Cybersecurity Expert
- GitHub: https://github.com/CyberGramps007
- Twitter: https://x.com/Grandpajoor
- Email: solaolabunmi@gmail.com
