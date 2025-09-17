Instructions to Run the Streamlit EDA App
1️⃣ Clone the repository

git clone https://github.com/your-username/streamlit-eda-app.git
cd streamlit-eda-app

2️⃣ Create a virtual environment (recommended)

python -m venv venv

Activate it:

Windows

venv\Scripts\activate

Mac/Linux

source venv/bin/activate

3️⃣ Install dependencies

pip install -r requirements.txt

4️⃣ Run the app

streamlit run app.py

5️⃣ Open in browser

After running, Streamlit will give you a local URL like:
http://localhost:8501
Open it in your browser to use the app.

⚡ How to use the app

Click "Upload a CSV file" and choose your dataset.

Explore:

Dataset preview

Shape & column info

Missing values

Summary statistics

Correlation heatmap

Column-wise analysis (Histogram, Boxplot, Bar chart)

Pairplot (relationships between numeric features)
