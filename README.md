\# Adaptive AI Cybersecurity Monitoring System



\## Project Overview



The Adaptive AI Cybersecurity Monitoring System is a machine-learning-based web application designed to monitor network traffic and detect potential cyber threats.



The system uses a Random Forest classification model trained using the CIC-IDS2017 dataset to identify benign traffic and different types of cyberattacks.



\## Key Features



\- Machine-learning-based network traffic classification.

\- Real-time network flow monitoring using PyCICFlowMeter.

\- Detection of multiple network attack categories.

\- Flask-based web application.

\- Interactive cybersecurity dashboard.

\- Prediction history and security alerts.

\- Administrator login and prediction review.

\- Feedback-driven adaptive learning.

\- Model version management and rollback.

\- SQLite database integration.



\## Technology Stack



| Component | Technology |

|---|---|

| Backend | Python, Flask |

| Frontend | HTML, CSS, JavaScript, Bootstrap |

| Machine Learning | Scikit-learn, Random Forest |

| Data Processing | Pandas, NumPy |

| Dataset | CIC-IDS2017 |

| Live Flow Capture | PyCICFlowMeter, Npcap |

| Database | SQLite |



\## Machine Learning Model



\- Active model: Random Forest (v2)

\- Input features: 77

\- Output classes: 15

\- Dataset: CIC-IDS2017



\## Project Structure



```text

project/

│

├── app.py

├── config.py

├── requirements.txt

├── README.md

├── .gitignore

│

├── src/

│   ├── adaptive.py

│   ├── data\_loader.py

│   ├── db.py

│   ├── evaluation.py

│   ├── live\_features.py

│   ├── live\_monitor.py

│   ├── model\_registry.py

│   └── repository.py

│

├── templates/

├── static/

├── models/

└── data/

```



\## Running the Application



1\. Clone the repository.

2\. Create and activate a Python virtual environment.

3\. Install the dependencies using `pip install -r requirements.txt`.

4\. Configure the required environment variables.

5\. Run the application using `python app.py`.

6\. Open `http://127.0.0.1:5000` in your browser.



\## Dataset



The project uses the CIC-IDS2017 dataset provided by the Canadian Institute for Cybersecurity.



Dataset source: https://www.unb.ca/cic/datasets/ids-2017.html



\## Current Implementation



The application includes Flask-based monitoring, Random Forest v2 classification, prediction logging, dashboard visualization, and live network flow capture.



Diagnostic sampled evaluation results should not be interpreted as independent benchmark results until training-data overlap has been ruled out.



\## Future Enhancements



\- Validate live detection using controlled attack traffic.

\- Improve feature compatibility between live flow extraction and training data.

\- Conduct independent model evaluation.

\- Enhance adaptive learning and monitoring capabilities.



\## Authors



\- K. Siddarth Varma

\- K. Bhumika

\- Gude Likhitha

\- Udgiri Satwika



\## Guide



Dr. Deevenaraju



Department of Computer Science and Engineering  

The ICFAI Foundation for Higher Education, Hyderabad



