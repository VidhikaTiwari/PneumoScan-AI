import { useLocation, useNavigate } from "react-router-dom";

function Results() {
  const location = useLocation();
  const navigate = useNavigate();

  const result = location.state?.result;
  const previewUrl = location.state?.previewUrl;

  if (!result) {
    return (
      <div className="results-page">
        <div className="result-card">
          <h2>No result found</h2>
          <p>Please upload and analyze an X-ray first.</p>
          <button onClick={() => navigate("/")}>Go Back</button>
        </div>
      </div>
    );
  }

  return (
    <div className="results-page">
      <div className="result-card">
        <h1>Scan Results</h1>

        {previewUrl && (
          <div className="result-image-container">
            <img src={previewUrl} alt="Uploaded X-ray" className="result-image" />
          </div>
        )}

        <p>
          <strong>Predicted Class:</strong> {result.predicted_class}
        </p>

        <p>
          <strong>Confidence:</strong> {(result.confidence * 100).toFixed(2)}%
        </p>

        <div className="probability-box">
          <h3>Class Probabilities</h3>
          <p>Normal: {(result.probabilities.NORMAL * 100).toFixed(2)}%</p>
          <p>Pneumonia: {(result.probabilities.PNEUMONIA * 100).toFixed(2)}%</p>
        </div>

        {result.warning && (
          <div className="warning-box">
            {result.warning}
          </div>
        )}

        <p className="disclaimer">
          For educational purposes only. This is not a medical diagnosis tool.
        </p>

        <button onClick={() => navigate("/")} className="back-btn">
          Analyze Another Scan
        </button>
      </div>
    </div>
  );
}

export default Results;