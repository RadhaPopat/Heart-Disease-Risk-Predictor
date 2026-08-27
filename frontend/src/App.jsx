import { useState } from "react";
import "./App.css";

function App() {
  const [formData, setFormData] = useState({
    age: 59,
    sex: 1,
    cp: 4,
    trestbps: 138,
    chol: 271,
    fbs: 0,
    restecg: 2,
    thalach: 182,
    exang: 0,
    oldpeak: 0,
    slope: 1,
    ca: 0,
    thal: 3,
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (e) => {
    const { name, value } = e.target;

    setFormData({
      ...formData,
      [name]: Number(value),
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.detail || "Unable to generate prediction."
        );
      }

      const data = await response.json();

      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>Cardiovascular Risk Assessment</h1>
          <p>
            AI-powered cardiovascular disease risk prediction with
            explainable results.
          </p>
        </div>
      </header>

      <main className="container">

        {/* Patient Form */}
        <section className="card">
          <h2>Patient Information</h2>
          <p className="section-description">
            Enter the patient's clinical information below.
          </p>

          <form onSubmit={handleSubmit}>
            <div className="form-grid">

              <div className="form-group">
                <label>Age</label>
                <input
                  type="number"
                  name="age"
                  value={formData.age}
                  onChange={handleChange}
                  min="1"
                  max="120"
                  required
                />
              </div>

              <div className="form-group">
                <label>Sex</label>
                <select
                  name="sex"
                  value={formData.sex}
                  onChange={handleChange}
                >
                  <option value="0">Female</option>
                  <option value="1">Male</option>
                </select>
              </div>

              <div className="form-group">
                <label>Chest Pain Type</label>
                <select
                  name="cp"
                  value={formData.cp}
                  onChange={handleChange}
                >
                  <option value="1">Typical Angina</option>
                  <option value="2">Atypical Angina</option>
                  <option value="3">Non-anginal Pain</option>
                  <option value="4">Asymptomatic</option>
                </select>
              </div>

              <div className="form-group">
                <label>Resting Blood Pressure</label>
                <input
                  type="number"
                  name="trestbps"
                  value={formData.trestbps}
                  onChange={handleChange}
                  required
                />
                <span className="unit">mmHg</span>
              </div>

              <div className="form-group">
                <label>Cholesterol</label>
                <input
                  type="number"
                  name="chol"
                  value={formData.chol}
                  onChange={handleChange}
                  required
                />
                <span className="unit">mg/dl</span>
              </div>

              <div className="form-group">
                <label>Fasting Blood Sugar</label>
                <select
                  name="fbs"
                  value={formData.fbs}
                  onChange={handleChange}
                >
                  <option value="0">≤ 120 mg/dl</option>
                  <option value="1">&gt; 120 mg/dl</option>
                </select>
              </div>

              <div className="form-group">
                <label>Resting ECG</label>
                <select
                  name="restecg"
                  value={formData.restecg}
                  onChange={handleChange}
                >
                  <option value="0">Normal</option>
                  <option value="1">ST-T Wave Abnormality</option>
                  <option value="2">
                    Left Ventricular Hypertrophy
                  </option>
                </select>
              </div>

              <div className="form-group">
                <label>Maximum Heart Rate</label>
                <input
                  type="number"
                  name="thalach"
                  value={formData.thalach}
                  onChange={handleChange}
                  required
                />
                <span className="unit">bpm</span>
              </div>

              <div className="form-group">
                <label>Exercise-Induced Angina</label>
                <select
                  name="exang"
                  value={formData.exang}
                  onChange={handleChange}
                >
                  <option value="0">No</option>
                  <option value="1">Yes</option>
                </select>
              </div>

              <div className="form-group">
                <label>ST Depression</label>
                <input
                  type="number"
                  step="0.1"
                  name="oldpeak"
                  value={formData.oldpeak}
                  onChange={handleChange}
                  required
                />
              </div>

              <div className="form-group">
                <label>ST Segment Slope</label>
                <select
                  name="slope"
                  value={formData.slope}
                  onChange={handleChange}
                >
                  <option value="1">Upsloping</option>
                  <option value="2">Flat</option>
                  <option value="3">Downsloping</option>
                </select>
              </div>

              <div className="form-group">
                <label>Major Vessels</label>
                <select
                  name="ca"
                  value={formData.ca}
                  onChange={handleChange}
                >
                  <option value="0">0</option>
                  <option value="1">1</option>
                  <option value="2">2</option>
                  <option value="3">3</option>
                </select>
              </div>

              <div className="form-group">
                <label>Thalassemia</label>
                <select
                  name="thal"
                  value={formData.thal}
                  onChange={handleChange}
                >
                  <option value="1">Normal</option>
                  <option value="2">Fixed Defect</option>
                  <option value="3">Reversible Defect</option>
                </select>
              </div>

            </div>

            <button
              type="submit"
              className="predict-button"
              disabled={loading}
            >
              {loading ? "Analyzing..." : "Predict Cardiovascular Risk"}
            </button>
          </form>
        </section>

        {/* Error */}
        {error && (
          <section className="error-card">
            <strong>Error:</strong> {error}
          </section>
        )}

        {/* Result */}
        {result && (
          <section className="card result-card">
            <h2>Risk Assessment</h2>

            <div className="risk-result">
              <div className="risk-percentage">
                {result.risk_percentage}%
              </div>

              <div className="risk-label">
                {result.risk_label}
              </div>
            </div>

            <p className="result-description">
              The model estimates the patient's cardiovascular risk
              probability at {result.risk_percentage}%.
            </p>

            <div className="prediction-details">
              <div>
                <span>Prediction</span>
                <strong>{result.prediction}</strong>
              </div>

              <div>
                <span>Probability</span>
                <strong>
                  {result.risk_probability}
                </strong>
              </div>
            </div>
          </section>
        )}

        {/* SHAP */}
        {result && result.feature_contributions && (
          <section className="card">
            <h2>Feature Contributions</h2>

            <p className="section-description">
              SHAP explains which patient features contributed
              towards increasing or decreasing the predicted risk.
            </p>

            <div className="contributions">

              {result.feature_contributions.map((item, index) => (
                <div
                  className="contribution"
                  key={index}
                >
                  <div className="contribution-info">
                    <strong>
                      {item.feature_name}
                    </strong>

                    <span>
                      {item.display_value}
                    </span>
                  </div>

                  <div className="contribution-bar-container">
                    <div
                      className={`contribution-bar ${
                        item.shap_value >= 0
                          ? "increase"
                          : "decrease"
                      }`}
                      style={{
                        width: `${Math.min(
                          Math.abs(item.shap_value) * 500,
                          100
                        )}%`,
                      }}
                    />
                  </div>

                  <div
                    className={`contribution-value ${
                      item.shap_value >= 0
                        ? "increase-text"
                        : "decrease-text"
                    }`}
                  >
                    {item.shap_value > 0 ? "+" : ""}
                    {item.shap_value.toFixed(4)}
                  </div>
                </div>
              ))}

            </div>
          </section>
        )}

      </main>

      <footer>
        <p>
          Cardiovascular Risk Prediction System
        </p>
        <p>
          Machine Learning • FastAPI • SHAP
        </p>
      </footer>
    </div>
  );
}

export default App;